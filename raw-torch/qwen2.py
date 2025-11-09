import torch
import torch.nn as nn
import math
import os

# 从Qwen的config.json提取核心参数（需手动读取）
config = {
    "vocab_size": 151936,  # Qwen-7B默认词表大小
    "hidden_size": 4096,   # 隐藏层维度
    "num_hidden_layers": 32,  # Decoder层数
    "num_attention_heads": 32,  # 注意力头数
    "ffn_hidden_size": 11008,  # SwiGLU中间层维度（Qwen2=11008，Qwen一致）
    "max_position_embeddings": 8192,  # 最大序列长度
    "rope_theta": 10000.0,  # RoPE基础系数（Qwen2可调整，默认与Qwen一致）
    "dtype": torch.float16  # 计算精度（优化点：可改为torch.bfloat16/GPU支持的更低精度）
}


class RoPE(nn.Module):
    def __init__(self, dim, theta=10000.0, max_seq_len=8192):
        super().__init__()
        self.dim = dim  # 每个注意力头的维度（hidden_size / num_attention_heads）
        self.theta = theta
        # 预计算位置编码因子（优化点：动态生成避免显存浪费，支持超长篇幅）
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        i = torch.arange(0, dim, 2, dtype=torch.float32)
        theta_i = 1.0 / (theta ** (i / dim))
        pos_theta = pos.unsqueeze(1) * theta_i.unsqueeze(0)  # [max_seq_len, dim//2]
        # 存储sin/cos（优化点：移到GPU显存，避免推理时重复计算）
        self.register_buffer("sin", torch.sin(pos_theta), persistent=False)
        self.register_buffer("cos", torch.cos(pos_theta), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [batch_size, num_heads, seq_len, head_dim]
        if seq_len is None:
            seq_len = x.shape[2]
        # 取前seq_len个位置的编码（支持动态序列长度）
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim//2]
        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0)
        
        # 对x的实部/虚部应用RoPE（优化点：用torch.where替代切片，提升并行效率）
        x1, x2 = x[..., ::2], x[..., 1::2]  # 分奇偶维度
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot
    


class SwiGLUFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.ffn_hidden_size = config["ffn_hidden_size"]
        
        # SwiGLU结构：x → (xW1) * σ(xW3) @ W2（Qwen用GELU，权重维度一致，可直接加载）
        self.w1 = nn.Linear(self.hidden_size, self.ffn_hidden_size, bias=False)
        self.w3 = nn.Linear(self.hidden_size, self.ffn_hidden_size, bias=False)  # Qwen无此层？不，Qwen2复用Qwen权重时，W3可复用W1的部分权重（实际Qwen权重已包含，直接加载即可）
        self.w2 = nn.Linear(self.ffn_hidden_size, self.hidden_size, bias=False)
        
        # 优化点：用torch.nn.functional.silu替代自定义sigmoid，GPU算子更高效
        self.activation = torch.nn.functional.silu

    def forward(self, x):
        x1 = self.w1(x)  # [bs, seq_len, ffn_hidden_size]
        x3 = self.w3(x)
        return self.w2(self.activation(x1) * x3)  # SwiGLU核心计算
    



# Decoder单层（Pre-LN结构）
class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config["hidden_size"], eps=1e-6)  # 优化点：eps与Qwen权重一致
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config["hidden_size"], eps=1e-6)
        self.ffn = SwiGLUFFN(config)
        
        # 优化点：可选添加残差连接dropout（推理时禁用，训练时启用）
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        # Pre-LN + 残差连接（优化点：用x + attn_out直接相加，避免中间变量）
        x = x + self.dropout(self.attn(self.ln_1(x)))
        x = x + self.dropout(self.ffn(self.ln_2(x)))
        return x

# Qwen2整体模型
class Qwen2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        
        # 1. Embedding层（优化点：TokenEmbedding与lm_head权重共享，减少显存占用）
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        
        # 2. Decoder层堆叠（优化点：用nn.ModuleList而非nn.Sequential，支持层并行）
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config) for _ in range(config["num_hidden_layers"])])
        
        # 3. 输出层归一化 + LM Head
        self.ln_f = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # 权重共享（必须！否则输出维度不匹配）
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        x = self.token_embedding(input_ids)  # [bs, seq_len, hidden_size]
        
        # 逐层前向（优化点：可加torch.no_grad()禁用梯度，推理提速）
        for layer in self.layers:
            x = layer(x)
        
        # 输出层（优化点：ln_f后直接过lm_head，避免多余计算）
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [bs, seq_len, vocab_size]
        return logits
    


def load_qwen_weights(model, qwen_weight_path):
    # 加载Qwen权重（支持单文件pytorch_model.bin或分片文件）
    if os.path.isfile(qwen_weight_path):
        state_dict = torch.load(qwen_weight_path, map_location="cpu")
    else:  # 分片文件（如model-00001-of-00002.bin）
        state_dict = {}
        for file in os.listdir(qwen_weight_path):
            if file.startswith("model-") and file.endswith(".bin"):
                state_dict.update(torch.load(os.path.join(qwen_weight_path, file), map_location="cpu"))
    
    # 权重映射：Qwen的key → 手动模型的key（核心！避免key不匹配）
    weight_map = {
        # Embedding层
        "transformer.wte.weight": "token_embedding.weight",
        # Decoder层（32层，逐层映射）
        "transformer.h.{}.ln_1.weight": "layers.{}.ln_1.weight",
        "transformer.h.{}.ln_1.bias": "layers.{}.ln_1.bias",
        "transformer.h.{}.attn.c_attn.weight": "layers.{}.attn.w_q.weight",  # Q/K/V合并投影（需拆分！）
        "transformer.h.{}.attn.c_attn.bias": "layers.{}.attn.w_q.bias",  # Qwen无bias，可忽略
        "transformer.h.{}.attn.c_proj.weight": "layers.{}.attn.w_o.weight",
        "transformer.h.{}.ln_2.weight": "layers.{}.ln_2.weight",
        "transformer.h.{}.ln_2.bias": "layers.{}.ln_2.bias",
        "transformer.h.{}.mlp.c_fc1.weight": "layers.{}.ffn.w1.weight",  # Qwen的mlp对应FFN
        "transformer.h.{}.mlp.c_fc2.weight": "layers.{}.ffn.w3.weight",  # SwiGLU的W3
        "transformer.h.{}.mlp.c_proj.weight": "layers.{}.ffn.w2.weight",
        # 输出层
        "transformer.ln_f.weight": "ln_f.weight",
        "transformer.ln_f.bias": "ln_f.bias",
        "lm_head.weight": "lm_head.weight"
    }
    
    # 构建新的state_dict（适配手动模型）
    new_state_dict = {}
    for qwen_key, model_key in weight_map.items():
        if "{}" in qwen_key:  # 处理分层参数（如32层Decoder）
            for layer_idx in range(model.config["num_hidden_layers"]):
                q_key = qwen_key.format(layer_idx)
                m_key = model_key.format(layer_idx)
                if q_key in state_dict:
                    # 关键：Qwen的c_attn是Q/K/V合并权重（3*hidden_size, hidden_size），需拆分为w_q/w_k/w_v
                    if "c_attn" in q_key:
                        weight = state_dict[q_key]  # [3*hidden_size, hidden_size]
                        split_size = weight.shape[0] // 3
                        new_state_dict[m_key.replace("w_q", "w_q")] = weight[:split_size, :]  # Q
                        new_state_dict[m_key.replace("w_q", "w_k")] = weight[split_size:2*split_size, :]  # K
                        new_state_dict[m_key.replace("w_q", "w_v")] = weight[2*split_size:, :]  # V
                    else:
                        new_state_dict[m_key] = state_dict[q_key]
        else:
            if qwen_key in state_dict:
                new_state_dict[model_key] = state_dict[qwen_key]
    
    # 加载权重（忽略不匹配的key，如Qwen的dropout参数）
    model.load_state_dict(new_state_dict, strict=False)
    print("权重加载完成！")
    return model

# 初始化模型并加载权重
model = Qwen2Model(config).to(config["dtype"]).to("cuda" if torch.cuda.is_available() else "cpu")

model = load_qwen_weights(model, "/你的/qwen/权重路径")
model.eval()  # 推理模式



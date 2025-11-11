import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from transformers import Qwen2ForCausalLM, AutoTokenizer
from safetensors.torch import load_file

# 1. 定义模型配置（与Qwen2-0.5B完全匹配）
qwen2_config = {
    "vocab_size": 151936,
    "hidden_size": 896,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "intermediate_size": 4864,
    "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "bos_token_id": 151643,
    "eos_token_id": 151643,
}

# ------------------------------
# 核心组件（与模型结构相关）
# ------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device = "cuda"):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x **2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float, device: torch.device) :
    inv_freq = 1.0 / (theta** (torch.arange(0, dim, 2, device=device) / dim))
    seq = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)
    
    concat_inv_freq = torch.concat([inv_freq, inv_freq], dim=-1)
    freqs = torch.outer(seq, concat_inv_freq)
    return torch.cos(freqs), torch.sin(freqs)
    

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x: torch.Tensor, position_embedding: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = position_embedding
    unsqueeze_dim = 1
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed  = (x * cos) + (rotate_half(x) * sin)
    return x_embed

class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, num_key_value_heads: int, device: torch.device):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, device=device)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, device=device)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, device=device)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False, device=device)

    def forward(self, hidden_states: torch.Tensor, position_embedding: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Q/K/V投影与多头拆分
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 应用RoPE
        q = apply_rope(q, position_embedding)
        k = apply_rope(k, position_embedding)

        # GQA扩展K/V头
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # 注意力计算
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim **0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ v

        # 合并多头并投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, device: torch.device):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device=device)
        self.up_proj = nn.Linear(hidden_size, intermediate_size,bias=False, device=device)
        self.down_proj = nn.Linear(intermediate_size, hidden_size,bias=False, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Block(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, num_key_value_heads: int, 
                 intermediate_size: int, rms_norm_eps: float, device: torch.device):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.attention = Attention(hidden_size, num_attention_heads, num_key_value_heads, device)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, device)

    def forward(self, hidden_states: torch.Tensor, position_embedding: torch.Tensor) -> torch.Tensor:
        # 注意力残差
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, position_embedding)
        hidden_states = residual + hidden_states

        # 前馈网络残差
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ------------------------------
# 推理专用模型类
# ------------------------------
class Qwen2InferenceModel(nn.Module):
    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.max_seq_len = config["max_position_embeddings"]
        self.bos_token_id = config["bos_token_id"]
        self.eos_token_id = config["eos_token_id"]

        # 模型组件（仅推理必需）
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, device=device)
        self.layers = nn.ModuleList([
            Qwen2Block(
                hidden_size=config["hidden_size"],
                num_attention_heads=config["num_attention_heads"],
                num_key_value_heads=config["num_key_value_heads"],
                intermediate_size=config["intermediate_size"],
                rms_norm_eps=config["rms_norm_eps"],
                device=device
            ) for _ in range(config["num_hidden_layers"])
        ])
        self.norm = RMSNorm(self.hidden_size, eps=config["rms_norm_eps"])
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, device=device, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # 共享词嵌入权重

        # 预计算RoPE频率（推理时直接复用）
        # self.register_buffer(
        #     "freqs_complex",
        #     precompute_rope_freqs(
        #         dim=self.hidden_size // config["num_attention_heads"],
        #         max_seq_len=self.max_seq_len,
        #         theta=config["rope_theta"],
        #         device=device
        #     ),
        #     persistent=False
        # )

        self.position_embedding = precompute_rope_freqs(
            dim=self.hidden_size // config["num_attention_heads"],
            max_seq_len=self.max_seq_len,
            theta=config["rope_theta"],
            device=device
        )

        # 推理模式：关闭dropout等训练相关操作
        self.eval()

    def forward(self, input_ids: torch.Tensor, output_hiddens=False):
        """单次前向传播，返回logits"""
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"输入长度({seq_len})超过最大序列长度({self.max_seq_len})")

        if output_hiddens:
            output_hidden_buffer = { "layers":[]}
        # 词嵌入
        hidden_states = self.embed_tokens(input_ids)

        if output_hiddens:
            output_hidden_buffer["embed_tokens"] = hidden_states

        # 应用所有Transformer层
        
        for layer in self.layers:
            cos, sin = self.position_embedding
            hidden_states = layer(hidden_states, (cos[:seq_len], sin[seq_len]))

            if output_hiddens:
                output_hidden_buffer["layers"].append(hidden_states)
            

        # 输出logits
        hidden_states = self.norm(hidden_states)

        if output_hiddens:
            output_hidden_buffer["logits"] = hidden_states

        if output_hiddens:
            return self.lm_head(hidden_states), output_hidden_buffer
 
        return self.lm_head(hidden_states)

    @torch.no_grad()  # 推理时禁用梯度计算
    def generate(self, prompt: str, tokenizer, max_new_tokens: int = 100, temperature: float = 1.0) -> str:
        """
        文本生成函数（贪心解码）
        :param prompt: 输入文本
        :param tokenizer: 分词器（需与模型匹配，如Qwen2Tokenizer）
        :param max_new_tokens: 最大生成token数
        :param temperature: 温度参数（控制随机性，0=贪心）
        :return: 生成的文本
        """
        # 分词并添加BOS
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)  # 转移到模型设备
        batch_size, current_len = input_ids.shape

        for _ in range(max_new_tokens):
            # 前向传播获取logits（仅需最后一个token的logits）
            logits = self.forward(input_ids)[:, -1, :]  # [batch, vocab_size]

            # 温度调整
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)  # 采样
            else:
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)  # 贪心

            # 拼接新token
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            current_len += 1

            # 检查是否生成EOS
            if next_token_id.item() == self.eos_token_id:
                break

            # 防止超出最大长度
            if current_len >= self.max_seq_len:
                break

        # 解码为文本
        return tokenizer.decode(input_ids[0], skip_special_tokens=True)

    @property
    def device(self) -> torch.device:
        return self.embed_tokens.weight.device

    def load_from_safetensors(self, weight_path = "/home/llmserver/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B/snapshots/91d2aff3f957f99e4c74c962f2f408dcc88a18d8/model.safetensors"):
        model_weightnames = self.state_dict().keys()
        loaded_weights = load_file(weight_path, device="cuda")
        converted_weights = {}
        for name, weight in loaded_weights.items():
            name = name.replace("model.", "")
            name = name.replace("self_attn", "attention")
            name = name.replace("mlp", "feed_forward")

            
            assert name in model_weightnames, f" weight not in model: {name}"

            converted_weights[name] = weight

        converted_weights["lm_head.weight"] = converted_weights["embed_tokens.weight"]
        self.load_state_dict(converted_weights)


# ------------------------------
# 推理示例
# ------------------------------
def generate(inputs: str|list[str]):

    # 2. 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 3. 加载模型（此处为随机权重示例，实际需加载预训练权重）
    model = Qwen2InferenceModel(qwen2_config, device)
    
    # 注意：实际使用时需加载预训练权重，例如：
    model_weightnames = model.state_dict().keys()
    model_path = "/home/llmserver/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B/snapshots/91d2aff3f957f99e4c74c962f2f408dcc88a18d8/model.safetensors"
    loaded_weights = load_file(model_path, device="cuda")
    converted_weights = {}
    for name, weight in loaded_weights.items():
        name = name.replace("model.", "")
        name = name.replace("self_attn", "attention")
        name = name.replace("mlp", "feed_forward")

        
        assert name in model_weightnames, f" weight not in model: {name}"

        converted_weights[name] = weight

    converted_weights["lm_head.weight"] = converted_weights["embed_tokens.weight"]
    model.load_state_dict(converted_weights)


    # 4. 加载分词器（需使用与Qwen2匹配的分词器）
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    # 确保分词器的特殊token与模型一致
    tokenizer.bos_token_id = qwen2_config["bos_token_id"]
    tokenizer.eos_token_id = qwen2_config["eos_token_id"]

    # 5. 文本生成示例
    # prompt = "Paris is the capital city of"
    # generated_text = model.generate(
    #     prompt=prompt,
    #     tokenizer=tokenizer,
    #     max_new_tokens=50,
    #     temperature=0.7
    # )
    # print(f"输入: {prompt}")
    # print(f"生成: {generated_text}")



# TODO
# def test_rope(my_model, ref_model:Qwen2ForCausalLM):
#     ref_embd = ref_model.model.layers[0].self_attn.rotary_emb
#     rotated = ref_embd(torch.arange(3, device=ref_model.device))

#     print(rotated)





def test_my_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 3. 加载模型（此处为随机权重示例，实际需加载预训练权重）
    model = Qwen2InferenceModel(qwen2_config, device)
    # model.load_from_safetensors()
    

    ref_model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
    input_ids = torch.tensor([[100, 200, 300]], device="cuda")
    
    outputs, hiddens = model(input_ids, output_hiddens=True)
    # ref_layer0 = ref_model.model.layers[0](ref_embed)

    # print("layer0 误差:", torch.norm(ref_layer0 - my_layer0).item())
    
    with torch.no_grad():
        outputs = ref_model(input_ids, output_hidden_states=True)
    hidden_embed = outputs.hidden_states[0]
    hidden_layer0 = outputs.hidden_states[1]

    my_embed = hiddens["embed_tokens"]
    my_layer0 = hiddens["layers"][0]

    print(hidden_layer0)
    
    print("Embedding误差:", torch.norm(hidden_embed - my_embed.to("cpu")).item())
    print("layer0误差:", torch.norm(hidden_layer0 - my_layer0.to("cpu")).item())

    # TODO
    # test_rope(my_model=model, ref_model=ref_model)

def run_ref_model():
    model_name = "Qwen/Qwen2-0.5B"
    model = Qwen2ForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "Paris is the capital city of"

    input_ids = tokenizer(prompt).input_ids
    input_ids = torch.tensor([input_ids, ])
    output_ids = model(input_ids)
    output_token = output_ids[:, -1:].argmax()

    # output_seq = output_ids
    output_seq = tokenizer.batch_decode(output_token)

    print(output_seq)
    


if __name__ == "__main__":
    test_my_model()
    
    # run_ref_model()
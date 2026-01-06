import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型名称或本地权重路径（替换为你的Qwen模型路径）
model_name_or_path = "Qwen/Qwen2.5-0.5B-Instruct"  # 或本地路径如"./qwen-7b"

# 加载模型（不加载Tokenizer也可，仅为演示）
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,  # 用FP16减少显存占用
    device_map="cpu",  # 加载到CPU，避免GPU显存不足
    trust_remote_code=True  # Qwen需要加载自定义代码
)

# 遍历state_dict，输出权重名称和形状
print(f"模型名称: {model_name_or_path}")
print(f"权重总数: {len(model.state_dict())}")
print("="*50)
for name, param in model.state_dict().items():
    print(f"name: {name}, shape: {param.shape}")



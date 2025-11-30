from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name =  "Qwen/Qwen2-0.5B"
model = AutoModelForCausalLM.from_pretrained( model_name)


tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = ["Paris is the capital city of"]


input_ids = tokenizer(inputs).input_ids
input_ids = torch.tensor(input_ids)
outputs = model.generate(input_ids)

outputs = tokenizer.batch_decode(outputs)

print(outputs)

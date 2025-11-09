from transformers import pipeline

model_name_or_path = "Qwen/Qwen3-8B"

generator = pipeline(
    "text-generation", 
    model_name_or_path, 
    torch_dtype="auto", 
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Give me a short introduction to large language models."},
]
messages = generator(messages, max_new_tokens=32768)[0]["generated_text"]
# print(messages[-1]["content"])

messages.append({"role": "user", "content": "In a single sentence."})
messages = generator(messages, max_new_tokens=32768)[0]["generated_text"]
# print(messages[-1]["content"])
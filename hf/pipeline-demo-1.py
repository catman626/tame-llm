from transformers import pipeline

model_name="Qwen/Qwen2-0.5B"
pipe = pipeline("text-generation", model=model_name)

prompt = ["Paris is the capital city of"]
outputs = pipe(prompt, max_length=20)

print(f" >>> output of pipeline is: {outputs}")

from transformers import pipeline

model_name_or_path = "Qwen/Qwen2-0.5B"

generator = pipeline(
    "text-generation", 

)
    # model_name_or_path, 
    # ="auto", 
# messages = [
    # {"role": "user", "content": "Paris is the capital city of"},
# ]
messages = generator(messages, max_new_tokens=20)[0]["generated_text"]
print(messages[-1]["content"])

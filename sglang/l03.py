from openai import OpenAI
import os
# Set OpenAI's API key and API base to use SGLang's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:30000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)



question = open(os.path.expanduser("~/inference/dataset/example-long.txt")).read()
chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        # {"role": "user", "content": "Paris is the capital city of"},
        {"role": "user", "content":question},
    ],
    max_tokens=8192,
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": True},
    },
)
print("Chat response:", chat_response)

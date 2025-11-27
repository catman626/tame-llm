from transformers import pipeline
from minference import MInference

if __name__ == "__main__":
    model_name="Qwen/Qwen2.5-7B-Instruct"
    pipe = pipeline("text-generation", model=model_name, torch_dtype="auto", device_map="auto")

    minference_patch = MInference("minference", model_name)
    pipe.model = minference_patch(pipe.model)

    prompt = ["Paris is the capital city of"]
    outputs = pipe(prompt, max_length=20)

    print(f" >>> output of pipeline is: {outputs}")


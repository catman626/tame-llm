# launch the offline engine
import sglang as sgl
import os

if __name__ == "__main__":
    # llm = sgl.Engine(model_path="facebook/opt-125m", context_length=100000)
    llm = sgl.Engine(model_path="Qwen/Qwen2.5-1.5B-Instruct", sliding_window=1000)

    # prompts = [
            # "Paris is the capital city of",
            # "Beijing is the capital city of"
    # ]

    
    with open(os.path.expanduser("~/inference/dataset/example-long.txt")) as f:
        prompts = [ f.read()] 

    outputs = llm.generate(prompts)


    for p, o in zip(prompts, outputs):
        print(f" >>> prompt: {p}, output: {o}")


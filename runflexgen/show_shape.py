import torch
import sys


if __name__ == "__main__":
    filename = sys.argv[1]
    
    if filename.endswith(".safetensors"):
        from safetensors.torch import load_file
        a = load_file(filename)
    else:
        a = torch.load(filename)

    if isinstance(a, torch.Tensor):
        print(a.shape)
    else:
        # print(a)
        # print(type(a))
        for name, data in a.items():
            print(f"{name}: {data.shape}")



import torch
import numpy as np

if __name__ == "__main__":
    a = torch.tensor([1, 2, 3], dtype=torch.bfloat16)
    b = a.numpy()

    print(b)

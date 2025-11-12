import torch
import os, sys


def compare(v1:torch.Tensor, v2:torch.Tensor):
    diff_l2_norm = torch.norm(v1 - v2).item()
    print(f" >>> difference: {diff_l2_norm}")


if __name__ == "__main__":
    my_value_file, golden_value_file = sys.argv[1], sys.argv[2]
    my_value = torch.load(my_value_file)
    golden_value = torch.load(golden_value_file)

    compare(my_value, golden_value)

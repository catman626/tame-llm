
import torch
import time


def cuda_create():
    context_size = 100000

    seqlens = torch.tensor([context_size], dtype=torch.int32, device="cuda")

    seqlens = torch.empty(1, dtype=torch.int32, device="cuda")
    seqlens[0] = context_size
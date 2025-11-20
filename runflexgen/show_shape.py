import torch
import sys


a = torch.load(sys.argv[1])
print(a.shape)
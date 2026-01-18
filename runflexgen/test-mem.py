import torch

def rms_layernorm(x, w):
    eps = 1e-6
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return w * (x / rms)

s = 100000
h = 1000
b = 8

hidden = torch.rand(b, s, h)
w = torch.rand(h, )
outputs = rms_layernorm(hidden, w)


print(f" >>> shape of outputs: {outputs.shape}")
mx = torch.cuda.max_memory_allocated() / 10** 9
print(f">>> max memory allocated: {mx} GB")
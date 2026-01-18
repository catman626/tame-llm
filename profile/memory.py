import torch
import time

n = 1e8
n = int(n)

a = torch.ones(n, device="cuda", dtype=torch.bfloat16)

b = a.to("cpu")

total = 0
repeat = 10
data_size = n *  2  # in bytes
for _ in range(repeat):
    st = time.time()
    b = a.to("cpu")

    elapse = time.time() - st
    total += elapse

avg = total / repeat

print(f" >>> avg transmission time: {avg} s")
print(f" >>> total transfer: {data_size / 10**9} GB")
print(f" >>> throughput: {data_size / avg / 10**9} GB/s")

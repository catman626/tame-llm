import torch
import ml_dtypes
import numpy as np
import time

def bf16_torch_to_np(torch_data):
    return torch_data.cpu().detach().view(torch.uint16).numpy().view(ml_dtypes.bfloat16)

def bf16_np_to_torch(np_data):
    return torch.tensor(np_data.view(np.uint16)).view(torch.bfloat16)


def make_test_data():
    b,h,s,d=1,28,100000,128
    k = s // 10
    h_kv=4

    data = torch.ones(b, h_kv, s, d, dtype=torch.bfloat16,device="cpu")
    idx = torch.randint(0, s, (b, h, k), device="cpu")
    rslt = torch.ones(b, h, k, d, dtype=torch.bfloat16, device="cpu")
    rslt = bf16_torch_to_np(rslt)
    
    return data, idx, rslt
    
if __name__ == "__main__":
    data, idx, rslt = make_test_data()

    repeat = 10
    total_time = 0
    for _ in range(repeat):
        d1, i1, r1 = bf16_torch_to_np(data), bf16_torch_to_np(idx), bf16_np_to_torch(rslt)   
    for _ in range(repeat):
        st = time.time()
        d1, i1, r1 = bf16_torch_to_np(data), bf16_torch_to_np(idx), bf16_np_to_torch(rslt)   
        elapse = time.time() - st
        total_time += elapse

    avg = total_time / repeat
    print(f" >>> average cost of convert: {avg}")

    
    

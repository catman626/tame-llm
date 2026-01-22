import torch
import time
# gather data(b,h,s,d) with idx(b,h,k)

def gather_func(data, idx):
    # data: (b, head, s, hidden)
    # idx:  (b, head, k)
    n_kvhead, head_dim = data.shape[1], data.shape[-1]
    n_qhead = idx.shape[1]
    selected = data.repeat_interleave(n_qhead//n_kvhead, dim=1)\
                    .gather(dim=-2, 
                            index=idx.unsqueeze(-1).expand(-1,-1,-1,head_dim))

    return selected

    
def single_test(data, idx, gather_func):
    total_elapse = 0
    repeat = 10
    for _ in range(repeat):
        st = time.time()
        g = gather_func(data, idx)
        elapse = time.time() - st
        total_elapse += elapse

    return total_elapse / repeat

def advanec_idx(data, idx):
    # concat all selected idx all 
    # data: (b, kv_head, s, hidden)
    # idx:  (b, q_head, k)
    b, kv_head, s, hidden = data.shape
    q_head = idx.shape[1]

    # 假设 q_head 是 kv_head 的整数倍（如 GQA）
    group_size = q_head // kv_head

    # 对每个 batch 和 q_head，映射到对应的 kv_head
    head_mapping = torch.arange(q_head, device=data.device) // group_size  # (q_head,)
    
    b_idx = torch.arange(b, device=data.device).view(b, 1, 1)        # (b, 1, 1)
    h_idx = head_mapping.view(1, q_head, 1)                         # (1, q_head, 1)
    s_idx = idx                                                     # (b, q_head, k)

    selected = data[b_idx, h_idx, s_idx, :]  # 自动广播，无需 expand

    return selected
    

def test_output(func1, func2):
    b,h,s,d=1,28,100000,128
    k = s // 10
    h_kv=4

    data = torch.ones(b,h_kv,s,d,dtype=torch.bfloat16, device="cuda")
    idx = torch.zeros(b,h,k, dtype=torch.int32, device="cuda")

    selected1 = func1(data, idx)
    selected2 = func2(data, idx)

    if torch.equal(selected1, selected2):
        print("[gather-based idx] and [advance-idx base idx] similar")
    else:
        print("idx funcs differ")
        
def make_data_idx(device):
    b,h,s,d=1,28,100000,128
    k = s // 10
    h_kv=4

    data = torch.ones(b,h_kv,s,d,dtype=torch.bfloat16, device=device)
    idx = torch.zeros(b,h,k, dtype=torch.int32, device=device)

    return data, idx

def test_time(idx_func):
    gpu_data, gpu_idx = make_data_idx("cuda")
    cpu_data, cpu_idx = make_data_idx("cpu")

    _ = single_test(gpu_data, gpu_idx, idx_func)
    _ = single_test(cpu_data, cpu_idx, idx_func)
    
    gpu_time = single_test(gpu_data, gpu_idx, idx_func)
    cpu_time = single_test(cpu_data, cpu_idx, idx_func)

    print(f" >>> gpu_time: {gpu_time} s")
    print(f" >>> cpu_time: {cpu_time} s")


if __name__ == "__main__":
    print(f" >>> gather")
    test_time(gather_func)
    
    print(" >>> advance idx")
    test_time(advanec_idx)

    test_output(gather_func, advanec_idx)
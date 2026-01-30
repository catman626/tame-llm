import torch
import time
import numpy as  np
import ml_dtypes
import dataclasses
# gather data(b,h,s,d) with idx(b,h,k)

from utils import TestDataConfig, block_size, make_kcache, make_block_idx

def gather_func(data, idx):
    # data: (b, head, s, hidden)
    # idx:  (b, head, k)
    n_kvhead, head_dim = data.shape[1], data.shape[-1]
    n_qhead = idx.shape[1]
    selected = data.repeat_interleave(n_qhead//n_kvhead, dim=1)\
                    .gather(dim=-2, 
                            index=idx.unsqueeze(-1).expand(-1,-1,-1,head_dim))

    return selected
    
def measure_time(gather_func, args):
    total_elapse = 0
    repeat = 10
    for _ in range(repeat):
        st = time.time()
        g = gather_func(*args)
        elapse = time.time() - st
        total_elapse += elapse

    return total_elapse / repeat

def indexing_blocks(data, idx):
    # data: (b, kv_head, s, hidden)
    # idx:  (b, q_head, k)
    s, b, n_kvhead, head_dim = data.shape
    _, q_head, n_select= idx.shape
    
    n_block = s // block_size()
    data = data.view(b, n_kvhead, n_block, -1)

    group_size = q_head // n_kvhead

    head_mapping = torch.arange(q_head, device=data.device) // group_size  # (q_head,)
    
    b_idx = torch.arange(b, device=data.device).view(b, 1, 1)        # (b, 1, 1)
    h_idx = head_mapping.view(1, q_head, 1)                         # (1, q_head, 1)
    s_idx = idx                                                     # (b, q_head, k)

    selected = data[b_idx, h_idx, s_idx]  # auto-broadcast, no need manual-expand

    return selected.view(b, q_head, n_select * block_size(), head_dim)

def dummy_slice_sbhd(data, idx):
    s, b, n_kvhead, d = data.shape
    n_qhead = idx.shape[1]
    group_size = n_qhead // n_kvhead

    k = idx.shape[2]
    return data[:k].clone()

def advance_idx_sbhd(data, idx):
    s, b, n_kvhead, d = data.shape
    n_qhead = idx.shape[1]
    group_size = n_qhead // n_kvhead
    
    s_idx = idx # (b, h, s)
    b_idx = torch.arange(b, device=data.device).view(b, 1, 1)
    head_idx= torch.arange(n_qhead, device=data.device)//group_size 
    head_idx = head_idx.view(1, -1, 1)
    return data[s_idx, b_idx, head_idx]

def dummy_flatten_idx_bhsd(data, idx):
    return torch.index_select(data.view(-1), 0, idx.view(-1))

def flatten_idx_sbhd(data, idx):
    # s, b, n_kvhead, d = data.shape
    return torch.index_select(data.view(-1), 0, idx.view(-1))

def indexing_via_take(data, idx):
    
    return torch.take(data, idx)


def loop_idxing(data, idx):
    # data: (b, kv_head, s, hidden)
    # idx:  (b, q_head, k)
    b, n_kvhead, s, hidden = data.shape
    n_qhead = idx.shape[1]
    k = idx.shape[2]
    group_size = n_qhead // n_kvhead

    rslt = torch.empty(b, n_qhead,k, hidden, device=data.device, dtype=data.dtype)
    for bno in  range(b):
        for hno in range(n_qhead):
            rslt[bno, hno] = torch.index_select(data[bno, hno//group_size], dim=0, index=idx[bno, hno, :])
    return rslt

def advance_idx(data, idx):
    # data: (b, kv_head, s, hidden)
    # idx:  (b, q_head, k)
    b, kv_head, s, hidden = data.shape
    q_head = idx.shape[1]
    group_size = q_head // kv_head

    b_idx = torch.arange(b, device=data.device).view(b, 1, 1)        # (b, 1, 1)

    head_mapping = torch.arange(q_head, device=data.device) // group_size  # (q_head,)
    h_idx = head_mapping.view(1, q_head, 1)                         # (1, q_head, 1)

    s_idx = idx                                                     # (b, q_head, k)

    selected = data[b_idx, h_idx, s_idx]  # auto-broadcast, no need manual-expand
    return selected

def bf16_torch_to_np(torch_data):
    return torch_data.cpu().detach().view(torch.uint16).numpy().view(ml_dtypes.bfloat16)

def bf16_np_to_torch(np_data):
    return torch.tensor(np_data.view(np.uint16)).view(torch.bfloat16)

def fancy_indexing(data, idx):
    data = bf16_torch_to_np(data)
    idx = idx.numpy()

    b, kv_head, s, hidden = data.shape
    q_head = idx.shape[1]

    # 假设 q_head 是 kv_head 的整数倍（如 GQA）
    group_size = q_head // kv_head

    # 对每个 batch 和 q_head，映射到对应的 kv_head
    head_mapping = np.arange(q_head) // group_size  # (q_head,)
    
    b_idx = np.arange(b).reshape(b, 1, 1)        # (b, 1, 1)
    h_idx = head_mapping.reshape(1, q_head, 1)                         # (1, q_head, 1)
    s_idx = idx                                                     # (b, q_head, k)

    selected = data[b_idx, h_idx, s_idx]  # 自动广播，无需 expand

    return bf16_np_to_torch(selected)

def test_output(func1, func2):
    b,h,s,d=1,28,100000,128
    k = s // 10
    h_kv=4

    data = torch.ones(b,h_kv,s,d,dtype=torch.bfloat16, device="cpu")
    idx = torch.zeros(b,h,k, dtype=torch.int32, device="cpu")

    selected1 = func1(data, idx)
    selected2 = func2(data, idx)

    if torch.equal(selected1, selected2):
        print("tested indexing function similar")
    else:
        print("idx funcs differ")
        

def make_data_idx(config:TestDataConfig):
    b, n_qhead, n_kvhead,s,d, sparsity = config.b, config.n_qhead, config.n_kvhead, config.s, config.head_dim, config.sparsity
    k = s // block_size() // sparsity * block_size()

    if config.layout == "bhsd":
        data = torch.ones(b,n_kvhead,s,d,dtype=torch.bfloat16, device=config.device)
    elif config.layout == "sbhd":
        data = torch.ones(s, b, n_kvhead , d, dtype=torch.bfloat16, device=config.device)
    else:
        raise NotImplementedError(f"invalid format: {config.layout}")
    idx = torch.randint(0, s, (b,n_qhead,k), dtype=torch.int32, device=config.device)

    return data, idx

def organize_test(func, args, test_tag):
    _ = measure_time(func, args)
    t = measure_time(func, args)
    print(f" >>> {test_tag} {t*1000:.3f} ms")

def organize_idx_test(idx_func, data, idx, test_tag):
    # data, idx = data_maker()

    # warmup
    _ = measure_time(idx_func, [data, idx], )
    
    t = measure_time(idx_func, [data, idx], )

    print(f" >>> idx:{list(idx.shape)}, data:{list(data.shape)}, {test_tag}: {t*1000:3f} ms")

def expand_idx(sbhd_data, idx):
    h_kv, d = sbhd_data.shape[2], sbhd_data.shape[-1]
    b, h, k = idx.shape
    s_block = b * h_kv * d
    b_block = h_kv * d
    h_block = d

    gqa_group = h // h_kv
    # print(f" >>> gqa_group: {gqa_group}")
    
    b_offset = torch.arange(b, device=sbhd_data.device).view(-1, 1,1) * b_block
    h_offset = (torch.arange(h, device=sbhd_data.device) // gqa_group) .view(1, -1, 1) * h_block 
    
    return (idx * s_block + b_offset + h_offset).view(-1)

def sparsity_test():
    config = TestDataConfig(layout="sbhd")
    data = make_kcache(config)

    def test_with_sparsity(data, sparsity):
        config = TestDataConfig(layout="sbhd", sparsity=sparsity)

        idx = make_block_idx(config)

        organize_idx_test(indexing_blocks, data, idx, 
                          f"indexing-block-(sparsity={sparsity})")
        
    for s in [5, 10, 20, 50, 100, 200]:
        test_with_sparsity(data, s)
    
    
if __name__ == "__main__":
    cpu_bhsd_cfg = TestDataConfig()
    cuda_bhsd_cfg = TestDataConfig(device="cuda")
    cpu_sbhd_cfg = TestDataConfig(layout="sbhd")
    
    cpu_bhsd_data, cpu_bhsd_idx= make_data_idx(cpu_bhsd_cfg)
    cuda_bhsd_data, cuda_bhsd_idx = make_data_idx(cuda_bhsd_cfg)
    cpu_sbhd_data, cpu_sbhd_idx = make_data_idx(cpu_sbhd_cfg)
    cuda_sbhd_data, cuda_sbhd_idx = make_data_idx(TestDataConfig(device="cuda", 
                                                                 layout="sbhd"))
    cpu_sbhd_block_idx = make_block_idx(cpu_sbhd_cfg)
    

    # print(f" >>> gather
    organize_idx_test(gather_func, cpu_bhsd_data, cpu_bhsd_idx , "cpu-gather")
    organize_idx_test(gather_func, cuda_bhsd_data, cuda_bhsd_idx,  "cuda-gather")
    
    # print(" >>> advance idx")
    organize_idx_test(advance_idx, cpu_bhsd_data, cpu_bhsd_idx, "cpu-fancy-indexing")
    organize_idx_test(advance_idx, cuda_bhsd_data, cuda_bhsd_idx, "cuda-fancy-indexing")

    organize_idx_test(fancy_indexing, cpu_bhsd_data, cpu_bhsd_idx, "numpy-fancy-indexing")

    # organize_idx_test(indexing_blocks, block_data_maker,"cpu-block-fancy-indexing")

    # sbhd_data_maker = make_data_idx("cpu", "sbhd")
    # organize_idx_test(advance_idx_sbhd, sbhd_data_maker, "cpu-indexing-sbhd-kvcache")

    organize_idx_test(dummy_slice_sbhd, cpu_bhsd_data, cpu_bhsd_idx, "cpu-slice-copy")

    organize_idx_test(dummy_flatten_idx_bhsd, cpu_bhsd_data, cpu_bhsd_idx, "dummy flattened")

    # organize_idx_test(loop_idxing, cpu_data_maker, "cpu-loop-indexing")

    organize_test(expand_idx, [cuda_sbhd_data, cuda_sbhd_idx] , "expand-idx")
   
    flatten_idx = expand_idx(cuda_sbhd_data, cuda_sbhd_idx).to("cpu")
    torch.cuda.synchronize()
    
    organize_idx_test(dummy_flatten_idx_bhsd, cpu_sbhd_data, cpu_sbhd_idx, "dummy flattened")
    organize_idx_test(dummy_flatten_idx_bhsd, cpu_sbhd_data, flatten_idx, "flatten")
    organize_idx_test(flatten_idx_sbhd, cpu_sbhd_data, flatten_idx, "flatten-2")

    take_idx = flatten_idx.unsqueeze(-1).expand(-1, TestDataConfig.head_dim)
    take_idx = take_idx + torch.arange(TestDataConfig.head_dim).view(1, -1)
    print(f" >>> shape of take_idx: {take_idx.shape}")

    organize_idx_test(indexing_via_take, cpu_sbhd_data, take_idx, "indexing-via-take")


    cache_config = TestDataConfig(layout="sbhd")
    cpu_sbhd_block_idx = make_block_idx(cache_config)
    cpu_sbhd_cache = make_kcache(cache_config)
    organize_idx_test(indexing_blocks, cpu_sbhd_cache, cpu_sbhd_block_idx, "indexing-block")

    selected = indexing_blocks(cpu_sbhd_cache, cpu_sbhd_block_idx)
    organize_test(torch.clone, [selected], "clone")

    sparsity_test()
    




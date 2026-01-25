import torch
import time
import numpy as  np
import ml_dtypes
import dataclasses
# gather data(b,h,s,d) with idx(b,h,k)

def block_size():
    return 64

@dataclasses.dataclass(frozen=True)
class TestConfig:
    b:int=4
    n_qhead:int=28
    n_kvhead:int=4
    s:int = 100000//block_size() *block_size()
    d:int = 128
    
def test_configs(indexing_block=False):
    b,n_qhead,s,d=2,28,100000,128
    n_kvhead = 4
    sparsity = 10
    s = s // block_size() * block_size()
    return b, (n_qhead, n_kvhead), s, d, sparsity

def gather_func(data, idx):
    # data: (b, head, s, hidden)
    # idx:  (b, head, k)
    n_kvhead, head_dim = data.shape[1], data.shape[-1]
    n_qhead = idx.shape[1]
    selected = data.repeat_interleave(n_qhead//n_kvhead, dim=1)\
                    .gather(dim=-2, 
                            index=idx.unsqueeze(-1).expand(-1,-1,-1,head_dim))

    return selected
    
def avg_time_test(data, idx, gather_func):
    total_elapse = 0
    repeat = 10
    for _ in range(repeat):
        st = time.time()
        g = gather_func(data, idx)
        elapse = time.time() - st
        total_elapse += elapse

    return total_elapse / repeat

def indexing_blocks(data, idx):
    # data: (b, kv_head, s, hidden)
    # idx:  (b, q_head, k)
    b, kv_head, s, hidden = data.shape
    _, q_head, n_select= idx.shape
    
    n_block = s // block_size()
    data = data.view(b, kv_head, n_block, -1)

    group_size = q_head // kv_head

    head_mapping = torch.arange(q_head, device=data.device) // group_size  # (q_head,)
    
    b_idx = torch.arange(b, device=data.device).view(b, 1, 1)        # (b, 1, 1)
    h_idx = head_mapping.view(1, q_head, 1)                         # (1, q_head, 1)
    s_idx = idx                                                     # (b, q_head, k)

    selected = data[b_idx, h_idx, s_idx]  # auto-broadcast, no need manual-expand

    return selected.view(b, q_head, n_select * block_size(), hidden)

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
        
def make_data_block_idx(device):
    def data_maker():
        b, (n_qhead, n_kvhead), s, hidden, sparsity = test_configs(indexing_block=True)
        n_block = s // block_size()
        n_select = n_block // sparsity
        
        data = torch.ones(b,n_kvhead,s,hidden,dtype=torch.bfloat16, device=device)
        idx = torch.randint(0, n_block, (b,n_qhead,n_select), dtype=torch.int32, device=device)

        return data, idx
    return data_maker

def make_data_idx(device, format="bhsd"):
    def data_maker():
        b,(h, h_kv),s,d, sparsity =test_configs()
        k = s // block_size() // sparsity * block_size()

        if format == "bhsd":
            data = torch.ones(b,h_kv,s,d,dtype=torch.bfloat16, device=device)
        elif format == "sbhd":
            data = torch.ones(s, b, h_kv , d, dtype=torch.bfloat16, device=device)
        idx = torch.randint(0, s, (b,h,k), dtype=torch.int32, device=device)

        return data, idx
    return data_maker

def organize_test(idx_func, data_maker, test_tag):
    data, idx = data_maker()

    _ = avg_time_test(data, idx, idx_func)
    
    t = avg_time_test(data, idx, idx_func)

    print(f" >>> idx:{list(idx.shape)}, data:{list(data.shape)}, {test_tag}: {t*1000:3f} ms")


if __name__ == "__main__":
    cuda_data_maker = make_data_idx("cuda")
    cpu_data_maker = make_data_idx("cpu")

    # print(f" >>> gather")
    organize_test(gather_func, cuda_data_maker, "cuda-gather")
    organize_test(gather_func, cpu_data_maker, "cpu-gather")
    
    # print(" >>> advance idx")
    organize_test(advance_idx, cuda_data_maker, "cuda-fancy-indexing")
    organize_test(advance_idx, cpu_data_maker, "cpu-fancy-indexing")

    organize_test(fancy_indexing, cpu_data_maker, "numpy-fancy-indexing")

    block_data_maker = make_data_block_idx("cpu")
    organize_test(indexing_blocks, block_data_maker,"cpu-block-fancy-indexing")

    sbhd_data_maker = make_data_idx("cpu", "sbhd")
    organize_test(advance_idx_sbhd, sbhd_data_maker, "cpu-indexing-sbhd-kvcache")

    organize_test(dummy_slice_sbhd, sbhd_data_maker, "cpu-slice-copy")

    organize_test(loop_idxing, cpu_data_maker, "cpu-loop-indexing")


    
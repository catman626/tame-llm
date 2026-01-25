import torch
import time
import numpy as  np
import ml_dtypes
import dataclasses

from utils import measure_time

@dataclasses.dataclass()
class TestConfig:
    b:int=2
    n_qhead:int=28
    n_kvhead:int=4
    block_size:int=64
    s:int = 100000//block_size*block_size
    hidden:int = 128
    sparsity :int= 10
    

def make_data_idx(test_config:TestConfig, device="cpu"):
    b,n_qhead, n_kvhead,s,d, sparsity = test_config.b, test_config.n_qhead, test_config.n_kvhead, test_config.s, test_config.hidden, test_config.sparsity

    k = s // TestConfig.block_size // sparsity * TestConfig.block_size

    data = torch.ones(b,n_kvhead,s,d,dtype=torch.bfloat16, device=device)
    idx = torch.randint(0, s, (b,n_qhead,k), dtype=torch.int32, device=device)

    return data, idx

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


def test_hidden_dim_impact():
    hidden_settings = [
        32,
        64,
        128,
        256,
        512,
    ]
    times = []
    for h in hidden_settings:
        cfg = TestConfig(hidden=h)
        data, idx = make_data_idx(cfg, "cpu")

        t = measure_time(advance_idx, [data, idx], repeat = 10)
        times.append(t)
        print(f" >>> hidden({h}): {t}s")


if __name__ == "__main__":
    test_hidden_dim_impact()
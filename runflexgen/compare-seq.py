import torch
import os, sys


def compare(v1:torch.Tensor, v2:torch.Tensor):
    # assert v1.shape == v2.shape, f" >>> shape different: {v1.shape} VS {v2.shape}"
    v1 = v1.cpu()
    v2 = v2.cpu()
    
    # (b, head, s, h)
    # v1 = v1.permute()
    # v1 = v1[:, :, -1]
    v2 = v2[:, :, :-2]

    diff_l2_norm = torch.norm(v1 - v2).item()
    print(f" >>> difference: {diff_l2_norm}")

def compare_seq(v1:torch.Tensor, v2:torch.Tensor):
    assert v1.shape == v2.shape, f" >>> shape different: {v1.shape} VS {v2.shape}"
    v1 = v1.cpu()
    v2 = v2.cpu()

    diff = v1 - v2
    # asser seq-dim == 2

    # (b, head, s, h) -> (s, b, head, h)
    diff = diff.permute(2, 0,1 ,3).flatten(start_dim=1)
    
    diff_l2_norm = torch.norm(diff, dim=-1)

    print(f" >>> diff: {diff_l2_norm}")


def index_into(t, seq_dim):
    if seq_dim == "1":
        ret = t[:, -1:]
    elif seq_dim == "2":
        ret = t[:, :, -1:]
    else:
        assert 0, f"invalid seq_dim: {seq_dim}"

    return ret

if __name__ == "__main__":
    my_value_file, golden_value_file = sys.argv[1], sys.argv[2]
    my_value = torch.load(my_value_file)
    golden_value = torch.load(golden_value_file)

    if len(sys.argv) >= 4:
        seq_dim_spec = sys.argv[3]
        if "-" in seq_dim_spec:
            seq_dims = seq_dim_spec.split('-')
            assert len(seq_dims) == 2
            # seq_dims = [int(d) for d in seq_dims]
            my_value = index_into(my_value, seq_dims[0])
            golden_value = index_into(my_value, seq_dims[1])

        else :
            
            golden_value = index_into(golden_value, seq_dim=seq_dim_spec)


    compare(my_value, golden_value)
    # compare_seq(my_value, golden_value)


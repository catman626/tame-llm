import torch
import matplotlib.pyplot as plt


import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def count_bos_is_max(attn_score:torch.Tensor):
    assert len(attn_score.shape) == 4
    
    max_idx = attn_score.argmax(-1)
    return (max_idx == 0).int().sum().item()

def plot_attention_distribution(attn_score, save_fname):
    # 1. 提取指定维度的数值
    # shape: (b, h, s, s) -> 取最后一个样本，最后一个头，最后一个token对所有token的关注
    # 注意：确保数据已经在 CPU 上并转为 numpy
    dist_data = attn_score[-1, -1, -1, :].detach().cpu().float().numpy()
    
    # 2. 绘图设置
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # 3. 绘制直方图和密度曲线 (KDE)
    # kde=True 会画出类似正态分布的那条平滑曲线
    sns.histplot(dist_data, kde=True, bins=50, color='royalblue', stat="density")
    
    # 4. 装饰图表
    plt.title("Distribution of Attention Scores (Last Token's View)", fontsize=14)
    plt.xlabel("Attention Score Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    
    # 计算一些统计指标标注在图上
    mean_val = np.mean(dist_data)
    std_val = np.std(dist_data)
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
    plt.legend()
    
    plt.savefig(save_fname)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def compute_attn_score(query, key):
    b, n_qhead, s, d = query.shape
    n_kvhead = key.shape[1]
    key_states = repeat_kv(key, n_qhead // n_kvhead)
    scaling = d ** -0.5
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    mask = torch.full((s, s), float('-inf'), dtype=query.dtype, device=query.device)
    mask = torch.triu(mask, diagonal=1)
    
    attn_weights = attn_weights + mask

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    
    return attn_weights

def plot(attn_weight, store_fname):
    assert len(attn_weight.shape) == 2
    assert not torch.isnan(attn_weight).any()

    plt.clf()

    processed = attn_weight.detach().cpu().numpy()

    mask = np.triu(np.ones_like(attn_weight, dtype=bool), k=1)
    
    processed[mask] = np.nan

    # processed = np.log(processed)
    plt.imshow(processed, cmap='viridis')
    plt.colorbar()
    plt.savefig(store_fname)

def make_attn_weight(layerno, rope):
    q_fname = f"dump/layer-{layerno}-query"
    k_fname = f"dump/layer-{layerno}-key"
    if rope:
        q_fname += "-rope"
        k_fname += "-rope"
    q = torch.load(q_fname)
    k = torch.load(k_fname)
    print(f" >>> shape of q: {q.shape}")
    print(f" >>> shape of k: {k.shape}")

    attn_weight = compute_attn_score(q, k)
    return attn_weight

def numerical_analysis(attn_weight):
    total_query = attn_weight.shape[1] * attn_weight.shape[2]  # n_head * n_q
    bos_is_max = count_bos_is_max(attn_weight)
    print(f" >>> total_query: {total_query}")
    print(f" >>> bos_is_max: {bos_is_max}")
    print(f" >>> ratio: {bos_is_max/total_query}")
    max_without_sink =  attn_weight[:, :, 1:, 1:].max().item()
    print(f" >>> max_without_sink: {max_without_sink}")
    max_idx = attn_weight[:, :, 1:, 1:].argmax()
    max_idx = torch.unravel_index(max_idx, attn_weight[:, :, 1:, 1:].shape)
    print(f" >>> max_idx: {max_idx}")

def block_pooling_attn_weight(attn_weight):
    b, h, sq, sk = attn_weight.shape
    assert sq == sk
    s = sq
    block_size = 128
    n_blk = sq // block_size
    s = n_blk * block_size
    attn_weight = attn_weight[:,:,:s, :s]
    
    full_block = torch.arange(s, dtype=torch.int).view(-1, 1) // block_size \
        >  torch.arange(s, dtype=torch.int).view(1, -1) // block_size

    denominator = full_block.int() * (block_size * (block_size -1)) + (block_size * (block_size+1))/2

    attn_weight = attn_weight.div(denominator)

    attn_weight = attn_weight.view(b, h, s//block_size, block_size, s // block_size, block_size)

    block_attn_weight = attn_weight.sum(dim=-1).sum(dim=-2)

    print(f" >>> shape of block_attn_weight: {block_attn_weight.shape}")

    return block_attn_weight

if __name__ == "__main__":
    fig_dir = "fig"
    layerno = 5
    
    attn_weight = make_attn_weight(5, rope=True)
    block_pooled_attn_score = block_pooling_attn_weight(attn_weight)

    numerical_analysis(attn_weight)

    # plot attn-weight
    for headno in range(14):
        plot(attn_weight[0, headno], f"{fig_dir}/{layerno}-{headno}")

    # plot attn-weight-after-block-pooling
    for headno in range(14):
        plot(attn_weight[0, headno], f"{fig_dir}/block-{layerno}-{headno}")


    # numerical_analysis of attn-score distribution
    plot_attention_distribution(attn_weight, f"{fig_dir}/{layerno}-distr")
    plot_attention_distribution(torch.log(attn_weight), f"{fig_dir}/{layerno}-log-distr")
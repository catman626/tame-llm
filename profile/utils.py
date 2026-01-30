import time
import dataclasses
import torch
def block_size():
    return 64

@dataclasses.dataclass(frozen=True)
class TestDataConfig:
    b:int=4
    n_qhead:int=28
    n_kvhead:int=4
    s:int = 100000//block_size() *block_size()
    head_dim:int = 128
    ffn_hidden_dim :int=4 * n_qhead * head_dim
    device:str = "cpu"
    layout :str = "bhsd"
    sparsity: int = 10
    
    def n_sample_block(self):
        return self.s // block_size() // self.sparsity

    def n_sample_token(self):
        return self.n_sample_block() * block_size()

    @property
    def hidden_dim(self):
        return self.n_qhead * self.head_dim
    
def make_kcache(config:TestDataConfig):
    if config.layout == "sbhd":
        return torch.rand(config.s, config.b, config.n_kvhead, config.head_dim, 
                      dtype=torch.bfloat16,
                      device=config.device)
    else:
        raise NotImplementedError(f" !!! make-cache for {config.layout} not implemented")

    
def make_block_idx(config:TestDataConfig):
    return torch.randint(0, config.s // block_size(), 
                         size=(config.b, config.n_qhead, config.n_sample_block()), 
                         device=config.device)

def measure_time(func, args, repeat=1):
    total_elapse = 0

    for _ in range(repeat):
        st = time.time()
        if isinstance(func, list):
            for f, a in zip(func, args):
                f(*a)
        else:
            func(*args)
        elapse = time.time() - st
        total_elapse += elapse
    
    avg = total_elapse / repeat
    return avg

def make_gated_up_down_projs_weight(config:TestDataConfig):
    dev = config.device
    up_s = 1
    up_proj = torch.rand(config.ffn_hidden_dim, config.hidden_dim, device=dev)
    gate_proj = torch.rand(config.ffn_hidden_dim, config.hidden_dim, device=dev)
    down_proj = torch.rand(config.hidden_dim, config.ffn_hidden_dim, device=dev)
    return up_proj, gate_proj, down_proj

def make_hidden(config:TestDataConfig):
    return torch.rand(config.b, config.s, config.n_qhead * config.head_dim, 
                    device=config.device)

def make_ln_weight(config:TestDataConfig):
    return torch.rand(config.b, config.s, config.n_qhead * config.head_dim,
                      device=config.device)

def make_mlp_weights(config:TestDataConfig):
    return *make_gated_up_down_projs_weight(config), make_ln_weight(config)
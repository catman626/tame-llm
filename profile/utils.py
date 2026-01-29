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
    device:str = "cpu"
    layout :str = "bhsd"
    sparsity: int = 10

    
    def n_sample_block(self):
        return self.s // block_size() // self.sparsity

    def n_sample_token(self):
        return self.n_sample_block() * block_size()
    
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

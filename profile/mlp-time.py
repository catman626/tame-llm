import torch
import torch.nn.functional as F
import dataclasses
import time

def block_size():
    return 64

@dataclasses.dataclass(frozen=True)
class TestConfig:
    b:int=4
    n_qhead:int=28
    n_kvhead:int=4
    s:int = 100000//block_size() *block_size()
    head_dim:int = 128
    ffn_hidden_dim:int = 4  * n_qhead * head_dim
    device:str = "cpu"
    dim_format :str = "bhsd"
    sparsity: int = 10
    

    @property
    def hidden_dim(self):
        return self.n_qhead * self.head_dim

    
def make_gated_up_down_projs_weight(config:TestConfig):
    dev = config.device
    up_s = 1
    up_proj = torch.rand(config.ffn_hidden_dim, config.hidden_dim, device=dev)
    gate_proj = torch.rand(config.ffn_hidden_dim, config.hidden_dim, device=dev)
    down_proj = torch.rand(config.hidden_dim, config.ffn_hidden_dim, device=dev)
    return up_proj, gate_proj, down_proj

def make_data_hidden(config:TestConfig):
    return torch.rand(config.b, config.s, config.n_qhead * config.head_dim, 
                      device=config.device)

def make_data_ln_weight(config:TestConfig):
    return torch.rand(config.b, config.s, config.n_qhead * config.head_dim,
                      device=config.device)

def rms_layernorm(x, w):
    eps = 1e-6
    rms = torch.mean(x * x, dim=-1, keepdim=True)
    rms = torch.sqrt(rms+eps)
    return w * (x / rms)




# @torch.compile(mode="reduce-overhead")
def qwen_mlp(inputs, w_gate, w_up, w_down, w_ln):
    # # decompress weights
    # if w_gate.device.device_type == DeviceType.COMPRESSED:
    #     w_gate = w_gate.device.decompress(w_gate)
    #     w_up = w_up.device.decompress(w_up)
    #     w_down = w_down.device.decompress(w_down)
    # (b, s, H) -> 8*10000*1000*2 -> 1.4G -> 5.6G

    norm = rms_layernorm(inputs, w_ln)

    gate = F.linear(norm, w_gate, bias=None)
    gate = F.silu(gate, inplace=True)
    up = F.linear(norm, w_up, bias=None)
    
    del norm

    out = up.mul_(gate)
    del gate
    
    out = F.linear(out, w_down, bias=None)
    
    mlp_out = out.add_(inputs)

    return mlp_out

def measure_time(gather_func, args):
    total_elapse = 0
    repeat = 10
    for _ in range(repeat):
        st = time.time()
        g = gather_func(*args)
        elapse = time.time() - st
        total_elapse += elapse

    return total_elapse / repeat
    
def organize_test(func, args, test_tag):
    _ = measure_time(func, args)
    t = measure_time(func, args)
    print(f" >>> {test_tag} {t*1000:.3f} ms")


def test_seq1_bs(bs, compile=False, log_memory_peak=False):
    config = TestConfig(b=bs, s=1,device="cuda")
    inputs = make_data_hidden(config)
    up_proj, gate_proj, down_proj = make_gated_up_down_projs_weight(config)
    ln_weight = make_data_ln_weight(config)
    
    mlp_func = qwen_mlp
    log = f"qwen_mlp-bs({bs})"
    if compile:
        mlp_func = torch.compile(mlp_func, mode="reduce-overhead")
        log = f"compiled-{log}"
    
    if log_memory_peak:
        torch.cuda.reset_max_memory_allocated()
        organize_test(mlp_func, [inputs, up_proj, gate_proj, down_proj, ln_weight], log)
        print(f" >>> mem_peak: {torch.cuda.max_memory_allocated()/ 1024**3:.3f}G")
    else:
        organize_test(mlp_func, [inputs, up_proj, gate_proj, down_proj, ln_weight], log)

    
if __name__ == "__main__":

    for i in range(10):
        test_seq1_bs(2**i)

    for i in range(10):
        test_seq1_bs(2**i, compile=True)

    
    
    
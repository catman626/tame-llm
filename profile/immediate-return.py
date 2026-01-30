import time
import torch
def make_up_down_projs_data(dev="cuda"):
    b,h,s,d=1,28,10000,128
    up_s = 1
    gpu_data = torch.rand(b, s, h*d, device=dev)
    up_proj = torch.rand(up_s* h*d, h*d, device=dev)
    down_proj = torch.rand(h*d, up_s*h*d, device=dev)
    return gpu_data, up_proj, down_proj

def measure_3tasks():
    gpu_data, up_proj, down_proj = make_up_down_projs_data("cuda")
    mm_stream = torch.cuda.Stream()

    with torch.cuda.stream(mm_stream):
        st = time.time()
        output = torch.nn.functional.linear(gpu_data, up_proj)
        output = torch.nn.functional.linear(output, down_proj)

    non_default_no_sync = time.time() - st
    mm_stream.synchronize()
    non_default_sync = time.time() - st
    
    st = time.time()
    output = torch.nn.functional.linear(gpu_data, up_proj)
    output = torch.nn.functional.linear(output, down_proj)
    default_nosync = time.time() - st
    torch.cuda.synchronize()
    default_sync = time.time() -st
    

    return default_nosync, default_sync, non_default_no_sync, non_default_sync 

def test_streams_ret_time():
    repeat = 10
    default_nosync, default_sync, non_default_no_sync, non_default_sync  = 0,0,0,0
    for _ in range(repeat):
        d, ds, nd, nds= measure_3tasks()
        default_nosync += d
        default_sync += ds
        non_default_no_sync += nd
        non_default_sync += nds
    
    default_nosync      = default_nosync        / repeat
    default_sync        = default_sync          / repeat
    non_default_no_sync = non_default_no_sync   / repeat
    non_default_sync    = non_default_sync      / repeat

    print(f" >>> default_nosync: {default_nosync}")
    print(f" >>> default_sync: {default_sync}")
    print(f" >>> non_default_no_sync: {non_default_no_sync}")
    print(f" >>> non_default_sync: {non_default_sync}")

if __name__ == "__main__":
    test_streams_ret_time()
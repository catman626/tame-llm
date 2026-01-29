import torch
import time
import threading
# gather data(b,h,s,d) with idx(b,h,k)
def warmup():
    gpu_data, up_proj, down_proj = make_up_down_projs_data("cuda")
    cpu_data, cpu_idx = make_data_idx("cpu")
    mm_stream = torch.cuda.Stream()
    
    # thread0: matmul
    # thread1: gather
    gather_worker(cpu_data, cpu_idx)   
    mm_worker(gpu_data, up_proj, down_proj, mm_stream)

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
        
def gather_worker(cpu_data, cpu_idx):
    with torch.profiler.record_function("gather-worker"):
        output = advanec_idx(cpu_data, cpu_idx)

def mm_worker(gpu_data, up_proj, down_proj, stream:torch.cuda.Stream, sync:bool=True):
    with torch.cuda.stream(stream), torch.profiler.record_function("mm-worker"):
        output = torch.nn.functional.linear(gpu_data, up_proj)
        output = torch.nn.functional.linear(output, down_proj)
    if sync:
        stream.synchronize()

def make_data_idx(device):
    b,h,s,d=1,28,100000,128
    k = s // 10
    h_kv=4

    data = torch.ones(b,h_kv,s,d,dtype=torch.bfloat16, device=device)
    idx = torch.zeros(b,h,k, dtype=torch.int32, device=device)

    return data, idx

def make_up_down_projs_data(dev="cuda"):
    b,h,s,d=1,28,10000,128
    up_s = 1
    gpu_data = torch.rand(b, s, h*d, device=dev)
    up_proj = torch.rand(up_s* h*d, h*d, device=dev)
    down_proj = torch.rand(h*d, up_s*h*d, device=dev)
    return gpu_data, up_proj, down_proj

def test_stream_overlap():
    gpu_data, up_proj, down_proj = make_up_down_projs_data("cuda")
    cpu_data, cpu_idx = make_data_idx("cpu")
    mm_stream = torch.cuda.Stream()

    st = time.time()
    with torch.cuda.stream(mm_stream):
        output = torch.nn.functional.linear(gpu_data, up_proj)
        output = torch.nn.functional.linear(output, down_proj)
    mm_stream.synchronize()
    
    gather_worker(cpu_data, cpu_idx)

    stream_overlap_time = time.time() - st

    print(f" >>> stream_overlap_time: {stream_overlap_time}")

def test_thread_overlap():
    gpu_data, up_proj, down_proj = make_up_down_projs_data("cuda")
    cpu_data, cpu_idx = make_data_idx("cpu")
    mm_stream = torch.cuda.Stream()
    
    # thread0: matmul
    # thread1: gather
    t_comp = threading.Thread(target=gather_worker, args=[cpu_data, cpu_idx])
    t_comm = threading.Thread(target=mm_worker, 
                              args=[gpu_data, up_proj, down_proj, mm_stream])
    
    with torch.profiler.profile(
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        st = time.time()
        t_comp.start()
        t_comm.start()
        
        t_comp.join()
        t_comm.join()
        
        elapse = time.time() - st
        print(f" >>> concurrent elapse: {elapse}")

    prof.export_chrome_trace("thread-overlap.json")

    st = time.time()
    mm_worker(gpu_data, up_proj, down_proj, mm_stream)
    gather_worker(cpu_data, cpu_idx)
    elapse = time.time() - st

    linear_time = measure_time([mm_worker, 
                                gather_worker] , 
                               [[gpu_data, up_proj, down_proj, mm_stream], 
                                [cpu_data, cpu_idx]])
    mm_time = measure_time(mm_worker, [gpu_data, up_proj, down_proj, mm_stream])
    gather_time = measure_time(gather_worker, [cpu_data, cpu_idx])
    
    print(f" >>> mm_time: {mm_time}")
    print(f" >>> gather_time: {gather_time}")
    print(f" >>> linear_time: {linear_time}")


def test_time(idx_func):
    gpu_data, gpu_idx = make_data_idx("cuda")
    cpu_data, cpu_idx = make_data_idx("cpu")

    _ = single_test(gpu_data, gpu_idx, idx_func)
    _ = single_test(cpu_data, cpu_idx, idx_func)
    
    gpu_time = single_test(gpu_data, gpu_idx, idx_func)
    cpu_time = single_test(cpu_data, cpu_idx, idx_func)

    print(f" >>> gpu_time: {gpu_time} s")
    print(f" >>> cpu_time: {cpu_time} s")

def test_sync():
    gpu_data, up_proj, down_proj = make_up_down_projs_data()
    stream = torch.cuda.Stream()
    sync_time = measure_time(mm_worker, [gpu_data, up_proj, down_proj, stream, True])
    async_time = measure_time(mm_worker, [gpu_data, up_proj, down_proj, stream, False])
    print(f" >>> async_time: {async_time}")
    print(f" >>> sync_time: {sync_time}")
    
if __name__ == "__main__":
    
    warmup()
    # test_thread_overlap()
    # test_stream_overlap()
    # d, up, down = make_up_down_projs_data()

    # mm_worker(d, up, down)
    test_sync()
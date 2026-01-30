import torch
import time
import threading
import importlib
mlp_time = importlib.import_module("mlp-time")
indexing_time = importlib.import_module("indexing-time")

from utils import measure_time, make_mlp_weights, make_hidden, TestDataConfig, \
    make_kcache, make_block_idx

def gather_func(data, idx):
    # data: (b, head, s, hidden)
    # idx:  (b, head, k)
    n_kvhead, head_dim = data.shape[1], data.shape[-1]
    n_qhead = idx.shape[1]
    selected = data.repeat_interleave(n_qhead//n_kvhead, dim=1)\
                    .gather(dim=-2, 
                            index=idx.unsqueeze(-1).expand(-1,-1,-1,head_dim))

    return selected

    

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
        
def indexing_worker(cpu_data, cpu_idx):
    with torch.profiler.record_function(f"gather-worker-tid({threading.get_native_id()})"):
        output = indexing_time.indexing_blocks(cpu_data, cpu_idx)
    
def mlp_worker(hidden, up_proj, gate_proj, down_proj, ln_weight, stream:torch.cuda.Stream):
    with torch.cuda.stream(stream), \
        torch.profiler.record_function(f"mlp-worker-tid({threading.get_native_id()})"):
        mlp_time.qwen_mlp(hidden, up_proj, gate_proj, down_proj, ln_weight)
        stream.synchronize()


def warmup(funcs, args):
    for f, a in zip(funcs, args):
        f(a)

def test_sequential(funcs, args, test_tag, dump_tracing_file):
    repeat = 10
    for _ in range(repeat):
        for f,a in zip(funcs, args):
            f(*a)
    
    total = 0
    with torch.profiler.profile(
        activities= [ torch.profiler.ProfilerActivity.CPU, 
                     torch.profiler.ProfilerActivity.CUDA ],
    ) as perf:
        for _ in range(repeat):
            st = time.time()
            for f,a in zip(funcs, args):
                f(*a)

            elapse = time.time() - st
            total += elapse
    perf.export_chrome_trace(dump_tracing_file)

    avg_elapse = total / repeat

    print(f" >>> {test_tag}: {avg_elapse*1000:.3f} ms")
        

def test_thread_overlap(funcs, args, test_tag, dump_tracing_file=None):
    # thread0: matmul
    # thread1: gather
    
    # warmup
    threads = []
    for f,a in zip(funcs, args):
        t = threading.Thread(target=f, args=a)
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    threads = []
    repeat = 10
    for _ in range(repeat):
        for f,a in zip(funcs, args):
            t = threading.Thread(target=f, args=a)
            threads.append(t)
    
    with torch.profiler.profile(
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
    
        st = time.time()
        for t in threads:
            t.start()

        for t in threads:
            t.join()
        
        elapse = time.time() - st
    elapse = elapse / repeat
    print(f" >>> {test_tag}: {elapse*1000:.3f} ms")

    if dump_tracing_file is not None:
        prof.export_chrome_trace(dump_tracing_file)

    
if __name__ == "__main__":
    # test_thread_overlap()
    # test_stream_overlap()
    # d, up, down = make_up_down_projs_data()

    # mm_worker(d, up, down)
    cuda_config = TestDataConfig(b=1, s=10000, device="cuda", layout="bshd")
    cpu_config = TestDataConfig(b=10,layout="sbhd")

    hidden = make_hidden(cuda_config)
    up_proj, gate_proj, down_proj, ln_weight= make_mlp_weights(cuda_config)
    mm_stream = torch.cuda.Stream()
    
    kcache = make_kcache(cpu_config)
    block_idx = make_block_idx(cpu_config)

    mlp_args = [ hidden, up_proj, gate_proj, down_proj, ln_weight, mm_stream]
    
    
    test_thread_overlap([indexing_worker, mlp_worker], args=[
        [ kcache, block_idx ],
        mlp_args,
    ], 
                        test_tag="thread-overlap",
                        dump_tracing_file="log/thread-overlap.json")  

    test_sequential([mlp_worker, indexing_worker], 
                    [   mlp_args,
                        [kcache, block_idx]],
                    test_tag="sequential run", 
                    dump_tracing_file="log/sequential.json")


    t_mlp = measure_time(mlp_worker, mlp_args)
    t_indexing = measure_time(indexing_worker, [kcache, block_idx])
    print(f" >>> t_mlp: {t_mlp*1000:.3f} ms")
    print(f" >>> t_indexing: {t_indexing*1000:.3f} ms")
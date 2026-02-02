import torch
import time
import threading
import importlib
mlp_time = importlib.import_module("mlp-time")
indexing_time = importlib.import_module("indexing-time")

from utils import measure_time, make_mlp_weights, make_hidden, TestDataConfig, \
    make_kcache, make_block_idx


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
        f(*a)

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
    warmup(funcs, args)
    
    with torch.profiler.profile(
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        threads = []
        repeat = 10
        for _ in range(repeat):
            for f,a in zip(funcs, args):
                t = threading.Thread(target=f, args=a)
                threads.append(t)
    
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

    t_mlp = measure_time(mlp_worker, mlp_args)
    t_indexing = measure_time(indexing_worker, [kcache, block_idx])
    print(f" >>> t_mlp: {t_mlp*1000:.3f} ms")
    print(f" >>> t_indexing: {t_indexing*1000:.3f} ms")
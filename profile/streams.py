import torch
from torch.profiler import profile, record_function, ProfilerActivity

device = torch.device('cuda')
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=0, warmup=0, active=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
) as prof:
    # stream s1
    with torch.cuda.stream(s1):
        with record_function("stream_s1"):
            z2 = y @ y

    # stream s2
    with torch.cuda.stream(s2):
        with record_function("stream_s2"):
            z3 = (x + y) @ (x - y)

    # 默认 stream
    with record_function("default_stream"):
        z1 = x @ x


    # 等待所有流完成（确保 profiler 能捕获完整事件）
    torch.cuda.synchronize()
    prof.step()
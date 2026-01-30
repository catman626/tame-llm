def test_sync():
    gpu_data, up_proj, down_proj = make_up_down_projs_data()
    stream = torch.cuda.Stream()
    sync_time = measure_time(mm_worker, [gpu_data, up_proj, down_proj, stream, True])
    async_time = measure_time(mm_worker, [gpu_data, up_proj, down_proj, stream, False])
    print(f" >>> without sync: {async_time}")
    print(f" >>> with sync: {sync_time}")
    print(f" >>> no-sync-case")
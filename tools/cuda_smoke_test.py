import time
import torch

def fmt_bytes(n: int) -> str:
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"

def main():
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        return
    dev_id = 0
    print("Device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(dev_id))

    free, total = torch.cuda.mem_get_info()
    print(f"VRAM free/total: {fmt_bytes(free)} / {fmt_bytes(total)}")

    # Sizes chosen to be friendly to small GPUs like MX550
    N = 2048
    dtype = torch.float16

    # Create pinned CPU tensors
    t0 = time.perf_counter()
    a_cpu = torch.randn((N, N), dtype=dtype, pin_memory=True)
    b_cpu = torch.randn((N, N), dtype=dtype, pin_memory=True)
    t1 = time.perf_counter()
    print(f"Host tensors created (pinned): {(t1 - t0)*1000:.2f} ms")

    stream = torch.cuda.Stream(device=dev_id)
    torch.cuda.synchronize()

    with torch.cuda.stream(stream):
        t2 = time.perf_counter()
        a = a_cpu.to(device=f"cuda:{dev_id}", non_blocking=True)
        b = b_cpu.to(device=f"cuda:{dev_id}", non_blocking=True)
        # Kick off matmul on the same stream
        c = a @ b
        t3 = time.perf_counter()
    # Wait for all to finish
    stream.synchronize()
    t4 = time.perf_counter()

    print(f"H2D + matmul (queued): {(t3 - t2)*1000:.2f} ms")
    print(f"Total H2D + matmul (sync): {(t4 - t2)*1000:.2f} ms")
    print("Result tensor:", c.dtype, c.device, c.shape)

    free2, total2 = torch.cuda.mem_get_info()
    print(f"VRAM free/total after op: {fmt_bytes(free2)} / {fmt_bytes(total2)}")

if __name__ == "__main__":
    main()

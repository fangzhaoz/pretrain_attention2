# spin_gpus.py
# Minimal GPU load: one process per GPU, BF16 GEMMs in a loop.
# Run:
#   python spin_gpus.py --nproc 8 --size 12288
#
# Stop:
#   Ctrl-C  (graceful), or
#   touch /tmp/STOP_SPIN   (if you pass --stop-file /tmp/STOP_SPIN), or
#   pkill -f spin_gpus.py  (force)
  
import argparse, os, time, signal, torch, torch.multiprocessing as mp
stop_evt = mp.Event()

def _sig_handler(*_):
    stop_evt.set()

def worker(rank: int, world_size: int, size: int, layers: int, stop_file: str | None):
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    g = torch.Generator(device=dev).manual_seed(1337 + rank)

    N = size
    # preallocate a couple of tiles to keep it simple
    A = torch.randn(N, N, dtype=torch.bfloat16, device=dev, generator=g)
    B = torch.randn(N, N, dtype=torch.bfloat16, device=dev, generator=g)
    C = torch.empty_like(A)

    # brief warmup
    for _ in range(3):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            C = A @ B
            A, B = C, A  # swap to vary operands a little
    torch.cuda.synchronize()

    # steady loop
    while not stop_evt.is_set():
        if stop_file and os.path.exists(stop_file):
            break
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            X, Y = A, B
            for _ in range(layers):
                C = X @ Y
                # light nonlinearity to keep kernels from being too repetitive
                X = torch.tanh(C)
                Y = C.transpose(0, 1)
        # tiny op to keep C "live"
        C.mul_(1.000001)
        # remove this sleep for even higher util; keep small to be responsive to Ctrl-C/stop-file
        time.sleep(0.001)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=None, help="GPUs to use (default: all visible)")
    ap.add_argument("--size", type=int, default=12288, help="Square GEMM size (e.g., 8192–16384)")
    ap.add_argument("--layers", type=int, default=6, help="Matmul depth per iteration")
    ap.add_argument("--stop-file", type=str, default=None, help="Optional path; create this file to stop")
    args = ap.parse_args()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _sig_handler)

    world_size = args.nproc or torch.cuda.device_count()
    assert world_size > 0, "No GPUs visible."

    mp.set_start_method("spawn", force=True)
    procs = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, args.size, args.layers, args.stop_file))
        p.daemon = False
        p.start()
        procs.append(p)

    try:
        # parent waits; Ctrl-C handled by signal -> stop_evt
        while any(p.is_alive() for p in procs):
            time.sleep(0.2)
            if stop_evt.is_set():
                break
    finally:
        # best-effort graceful shutdown
        stop_evt.set()
        for p in procs:
            p.join(timeout=5)

if __name__ == "__main__":
    main()

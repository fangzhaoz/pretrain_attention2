#!/usr/bin/env python3
# polite_babysitter_multi.py
# Keeps GPUs above a util floor without colliding with your real jobs.
# - Multi-GPU, process-aware (skips if another PID uses >= proc_mem_floor_mib)
# - Idle grace window so it won't run during short gaps between your loops
# - Early-abort during a burst if memory pressure rises
# - Per-process VRAM cap (best-effort)
# - Uses nvidia-smi; no pynvml dependency

import argparse, math, os, signal, subprocess, sys, threading, time
import torch

stop_flag = False
def _sig_handler(sig, frame):
    global stop_flag
    stop_flag = True
    print("\n[manager] stopping…", flush=True)

for s in (signal.SIGINT, signal.SIGTERM):
    signal.signal(s, _sig_handler)

def sh(cmd):
    return subprocess.check_output(cmd, text=True).strip()

def get_gpu_count():
    out = sh(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"])
    return 0 if not out else len(out.splitlines())

def gpu_util_free_used(gpu):
    # returns (util%, free_MiB, used_MiB)
    out = sh([
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.free,memory.used",
        "--format=csv,noheader,nounits", "-i", str(gpu)
    ])
    util_s, free_s, used_s = [t.strip() for t in out.split(",")]
    return int(util_s), int(free_s), int(used_s)

def gpu_active_processes(gpu):
    """
    Returns list of (pid:int, used_mib:int) for compute apps on this GPU.
    """
    try:
        out = sh([
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits", "-i", str(gpu)
        ])
        lines = [l for l in out.splitlines() if l.strip()]
        procs = []
        for l in lines:
            pid_s, mem_s = [t.strip() for t in l.split(",")]
            procs.append((int(pid_s), int(mem_s)))
        return procs
    except subprocess.CalledProcessError:
        return []

def choose_N(free_bytes, dtype=torch.bfloat16, safety_frac=0.20):
    # Use at most a fraction of *free* VRAM for GEMM tensors
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    budget = max(int(free_bytes * safety_frac), 64 * 1024 * 1024)  # >=64 MiB
    N = int(math.sqrt(budget / (3 * bytes_per)))  # A(NxN),B(NxN),C(NxN)
    return max(512, (N // 128) * 128)  # align for tensor cores

@torch.inference_mode()
def polite_burst(device, seconds, N, used_mib_baseline, dtype=torch.bfloat16):
    # Low-priority stream so real work preempts this compute.
    low, high = torch.cuda.get_stream_priority_range()
    stream = torch.cuda.Stream(device=device, priority=low)

    with torch.cuda.stream(stream):
        A = torch.randn((N, N), device=device, dtype=dtype)
        B = torch.randn((N, N), device=device, dtype=dtype)
        C = torch.empty((N, N), device=device, dtype=dtype)
        end_t, iters, last_check = time.time() + seconds, 0, 0.0
        while time.time() < end_t:
            torch.matmul(A, B, out=C)
            # Tiny tweaks to prevent constant-fold/over-fusion no-ops
            A.add_(0.0009765625)  # +2^-10
            B.t_()
            iters += 1
            # Poll ~1 Hz; abort if used memory rises (another job started)
            now = time.time()
            if now - last_check >= 1.0:
                last_check = now
                _, _, used_mib_now = gpu_util_free_used(device)
                if used_mib_now > used_mib_baseline + 256:  # 256 MiB slack
                    break

    # Free immediately
    del A, B, C
    torch.cuda.empty_cache()
    stream.synchronize()
    return iters

def babysit_gpu(gpu, args, self_pid):
    torch.cuda.set_device(gpu)
    # Best-effort cap so this process can't hoard VRAM
    try:
        torch.cuda.set_per_process_memory_fraction(args.self_vram_cap, device=gpu)
    except Exception:
        pass  # not available everywhere; other guards still protect us

    print(f"[GPU{gpu}] watching (th={args.util_threshold}%, burst={args.burst_seconds}s, "
          f"proc_floor={args.proc_mem_floor_mib} MiB, idle_grace={args.idle_grace}s)", flush=True)

    low_since = None  # when util first dropped below threshold with no big process

    while not stop_flag:
        try:
            util, free_mib, used_mib = gpu_util_free_used(gpu)
            procs = [(pid, mem) for (pid, mem) in gpu_active_processes(gpu) if pid != self_pid]
            someone_big = any(mem >= args.proc_mem_floor_mib for _, mem in procs)

            # If util high or another big process present, reset grace timer.
            if util >= args.util_threshold or someone_big:
                low_since = None
                time.sleep(args.check_interval)
                continue

            # Util is low and no big process. Start/keep grace window.
            now = time.time()
            if low_since is None:
                low_since = now
                time.sleep(args.check_interval)
                continue

            if now - low_since < args.idle_grace:
                time.sleep(args.check_interval)
                continue

            # Past grace window; consider a burst if enough free memory.
            if free_mib >= args.min_free_mb:
                # Baseline used memory to detect new activity during the burst
                _, _, used_mib0 = gpu_util_free_used(gpu)
                N = choose_N(free_mib * 1024 * 1024, dtype=torch.bfloat16, safety_frac=args.safety_frac)
                try:
                    t0 = time.time()
                    iters = polite_burst(gpu, args.burst_seconds, N, used_mib0, dtype=torch.bfloat16)
                    print(f"[GPU{gpu}] burst N={N}, iters={iters}, {time.time()-t0:.1f}s "
                          f"(util was {util}%, free {free_mib} MiB)", flush=True)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"[GPU{gpu}] OOM; backing off 1h…", flush=True)
                        torch.cuda.empty_cache()
                        for _ in range(3600):
                            if stop_flag: break
                            time.sleep(1)
                    else:
                        print(f"[GPU{gpu}] runtime error: {e}", flush=True)
                        time.sleep(args.check_interval)
            else:
                time.sleep(args.check_interval)

        except Exception as e:
            print(f"[GPU{gpu}] loop exception: {e}", flush=True)
            time.sleep(args.check_interval)

def parse_gpu_list(arg, total):
    if arg.strip().lower() in ("all", "*"):
        return list(range(total))
    out = []
    for p in arg.split(","):
        p = p.strip()
        if "-" in p:
            a, b = p.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(p))
    return sorted(set([g for g in out if 0 <= g < total]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=str, default="all", help="e.g. 'all', '0-7', or '0,2,4,6'")
    ap.add_argument("--util-threshold", type=int, default=25, help="min utilization %% to keep")
    ap.add_argument("--check-interval", type=int, default=15, help="seconds between checks")
    ap.add_argument("--idle-grace", type=int, default=180, help="util must stay low this many seconds before burning")
    ap.add_argument("--burst-seconds", type=int, default=120, help="seconds of compute per burst")
    ap.add_argument("--min-free-mb", type=int, default=512, help="skip if free MiB below this")
    ap.add_argument("--safety-frac", type=float, default=0.20, help="fraction of *free* VRAM used for GEMM tensors")
    ap.add_argument("--proc_mem_floor_mib", type=int, default=1024, help="treat any other PID >= this as active job")
    ap.add_argument("--self_vram_cap", type=float, default=0.05, help="per-process VRAM cap (0..1), best-effort")
    args = ap.parse_args()

    # Discover GPUs
    try:
        n = get_gpu_count()
    except Exception as e:
        print(f"[manager] nvidia-smi error: {e}")
        sys.exit(1)

    sel = parse_gpu_list(args.gpus, n)
    if not sel:
        print("[manager] no GPUs selected")
        sys.exit(1)

    print(f"[manager] Found {n} GPUs. Managing: {sel}", flush=True)

    self_pid = os.getpid()
    threads = []
    for g in sel:
        t = threading.Thread(target=babysit_gpu, args=(g, args, self_pid), daemon=True)
        t.start()
        threads.append(t)

    try:
        while not stop_flag:
            time.sleep(1)
    finally:
        for t in threads:
            t.join(timeout=2.0)
        print("[manager] Clean exit.", flush=True)

if __name__ == "__main__":
    main()

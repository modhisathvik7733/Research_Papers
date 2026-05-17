"""Auto-imported (cwd = SMF repo root is on sys.path) by every python
invocation in run_0010.sh — retrofit, sparse, eval. Enables A100 fast
paths globally without forking the SMF source."""
try:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True      # A100 TF32 matmul
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True             # static shapes -> faster
    try:
        torch.set_float32_matmul_precision("high")    # TF32-class fp32 matmul
    except Exception:
        pass
    if torch.cuda.is_available():
        print(f"[sitecustomize] A100 fast paths ON "
              f"(TF32+matmul-high+cudnn.benchmark) | "
              f"{torch.cuda.get_device_name(0)}")
except Exception as e:                                  # never block the run
    print("[sitecustomize] torch perf setup skipped:", e)

#!/usr/bin/env python
"""Top-level trainer entry for R2-Dreamer x Clash Royale.

Runs ``r2dreamer/train.py`` (a Hydra app) from the repo root, no ``cd``
required.  Defaults mirror ``r2dreamer/runs/cr.sh``:

    python train_cr.py                  # CPU, 1 env, 500k steps
    python train_cr.py 2000             # CPU, custom steps
    python train_cr.py --gpu 2000       # GPU (cuda) + 8 parallel CPU sim workers
    python train_cr.py --gpu --env-num 16 --compile 200000

Architecture: the simulator always runs in CPU subprocesses (ParallelEnv
workers); only agent inference + world-model updates run on the device
(--gpu moves those to cuda, pin_memory + async H2D handled upstream).
"""
import argparse
import datetime
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_R2D = os.path.join(_HERE, "r2dreamer")


def _parse_int(s):
    """Accept plain ints and scientific notation like 2e5, 5e5 (as Hydra does)."""
    try:
        return int(s)
    except ValueError:
        return int(float(s))


def main():
    ap = argparse.ArgumentParser(
        description="Train R2-Dreamer on the Clash Royale simulator (repo root entry)."
    )
    ap.add_argument("steps", nargs="?", type=_parse_int, default=500000,
                    help="total env steps (default 500000; supports 2e5)")
    ap.add_argument("--gpu", action="store_true",
                    help="train on cuda with --env-num parallel CPU sim workers")
    ap.add_argument("--env-num", type=int, default=8,
                    help="parallel sim workers (GPU mode only)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the update fn (GPU mode only; needs triton)")
    ap.add_argument("--logdir", default=None,
                    help="override logdir (default logdir/<MMDD>_r2dreamer_cr)")
    ap.add_argument("--max-size", type=_parse_int, default=None,
                    help="replay buffer capacity in steps (default steps+10000; supports 3e5)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the train.py command and exit without running")
    args = ap.parse_args()

    date = datetime.datetime.now().strftime("%m%d")
    max_size = args.max_size or (args.steps + 10000)
    overrides = [
        "env=cr",
        f"env.steps={args.steps}",
        "model.rep_loss=r2dreamer",
        # Replay buffer on CPU: CR grids are ~35 KB/step, the GPU is only
        # 24 GB, and pre-allocating the buffer there OOMs (sample() transfers
        # each batch back to the device via pin_memory + non_blocking).
        "buffer.storage_device=cpu",
        f"buffer.max_size={max_size}",
        "batch_size=16",
        "batch_length=64",
        "trainer.train_ratio=64",
        "seed=0",
    ]
    if args.gpu:
        overrides += [
            "device=cuda",
            f"env.env_num={args.env_num}",
            f"model.compile={str(args.compile).lower()}",
        ]
    else:
        overrides += ["device=cpu", "model.compile=False", "env.env_num=1"]
    overrides.append("logdir=" + (args.logdir or f"logdir/{date}_r2dreamer_cr"))

    # CPU updates use bf16 autocast; oneDNN's bf16 backward crashes on
    # avx2_vnni_2 CPUs ("DNNL does not support bf16/f16 backward"). Cap ISA.
    # Harmless on GPU (fp16 via cuDNN there).
    os.environ.setdefault("ONEDNN_MAX_CPU_ISA", "AVX2")

    print(f"[train_cr] cwd -> {_R2D}")
    print(f"[train_cr] python train.py {' '.join(overrides)}")
    if args.dry_run:
        sys.exit(0)
    sys.exit(subprocess.call([sys.executable, "train.py", *overrides], cwd=_R2D))


if __name__ == "__main__":
    main()

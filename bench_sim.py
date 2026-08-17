"""Benchmark the CREnv simulator speed: single env vs 8 parallel envs.

Run: python bench_sim.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r2dreamer'))
import numpy as np
import torch


def random_action():
    a = np.zeros((5, 32, 18), dtype=np.float32)
    for seg in (a[0], a[1], a[2]):
        seg[np.random.randint(len(seg))] = 1.0
    return a


def bench_single(n=300):
    from envs.cr import ClashRoyale
    env = ClashRoyale()
    env.reset()
    times = []
    ents = []
    for i in range(n):
        t0 = time.time()
        o, r, d, _ = env.step(random_action())
        times.append(time.time() - t0)
        ents.append(int((o['grid'] > 0).any(axis=-1).sum()))
        if d:
            env.reset()
    times = np.array(times)
    print(f"[single env] {n} steps: avg {times.mean()*1000:.1f} ms/step "
          f"(p50 {np.median(times)*1000:.1f}, p90 {np.percentile(times,90)*1000:.1f}, max {times.max()*1000:.1f})")
    print(f"[single env] avg occupied cells: {np.mean(ents):.1f}")


def bench_parallel(n_envs=8, n_rounds=100):
    from envs.parallel import ParallelEnv
    from envs.cr import ClashRoyale

    def constructor(i):
        return lambda: ClashRoyale(seed=i)

    pe = ParallelEnv(constructor, n_envs, 'cpu')
    done = torch.ones(n_envs, dtype=torch.bool)
    t0 = time.time()
    for _ in range(n_rounds):
        act = np.zeros((n_envs, 55), dtype=np.float32)
        for b in range(n_envs):
            a = np.zeros(55, dtype=np.float32)
            for lo, hi in [(0, 5), (5, 37), (37, 55)]:
                a[lo + np.random.randint(hi - lo)] = 1.0
            act[b] = a
        trans, done = pe.step(torch.as_tensor(act), done)
    dt = time.time() - t0
    total = n_rounds * n_envs
    print(f"[{n_envs} parallel envs] {n_rounds} rounds: {dt:.1f}s wall "
          f"-> {dt/n_rounds*1000:.0f} ms/round, {total/dt:.1f} steps/s (fps), {total/dt/n_envs:.1f} fps per env")


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if which in ('single', 'both'):
        bench_single()
    if which in ('parallel', 'both'):
        bench_parallel()

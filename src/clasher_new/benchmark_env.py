"""Measure PPO rollout throughput of the simulator as a function of env count.

The Clash Royale simulator is pure Python (GIL-bound), so the only way to
parallelise env stepping is multiple processes (``SubprocVecEnv``). This
script measures the throughput of the *real training loop* — batched policy
inference on the main process + parallel env stepping + per-episode opponent
sampling — for a range of ``n_envs`` values, and prints a table.

Usage (from ``src/clasher_new``):

    python benchmark_env.py            # sweep K = 1,2,4,...,24
    python benchmark_env.py --k 8      # a single K
    python benchmark_env.py --budget 15

The goal is to find the K (number of parallel envs) that best uses the
machine's cores. On a 24-thread CPU you typically see throughput saturate
when the *main-process* policy inference (which grows with K) stops being
cheaper than the per-env sim step.

Note: stable-baselines3's ``SubprocVecEnv`` defaults to the *forkserver*
start method, which re-imports the ``__main__`` module from its file path.
That is why this must be run as a real file, not a heredoc.
"""
import argparse
import os
import tempfile
import time

import numpy as np
import torch
from stable_baselines3 import PPO

from environment import random_strategy
from self_play import (
    OpponentPool,
    make_opponent_vec_env,
    bridge_rush_left_script,
    bridge_rush_script,
    defender_script,
)

_BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_BASE)

# Main-process policy inference is fastest single-threaded for this small
# model; more threads just fight each other on a 0.62M-param forward pass.
torch.set_num_threads(1)

_MODEL = os.path.join(_BASE, 'cr_discrete.zip')


# ---------------------------------------------------------------------------
# Per-env opponent sampling (the architecture being benchmarked)
# ---------------------------------------------------------------------------

def _make_pool(seed):
    # Empty recent_dir so the "recent" bucket is skipped, leaving a ~50/50
    # mix of a real PPO model opponent and scripted baselines — the same
    # compute profile the real training loop sees.
    return OpponentPool(
        base_checkpoints=[_MODEL],
        recent_dir=os.path.join(tempfile.mkdtemp(), 'cr_logs'),
        prefix='cr',
        n_recent=6,
        fixed_strategies=[random_strategy, bridge_rush_script,
                          bridge_rush_left_script, defender_script],
        weights={'recent': 0.6, 'base': 0.2, 'fixed': 0.2},
        device='cpu',
        seed=seed,
    )


def make_vec(n_envs, seed=1234):
    # The real train.py factory: each sub-env re-picks its opponent from its
    # own ``pool.child(rank)`` at every episode reset.
    return make_opponent_vec_env(_make_pool(seed), n_envs=n_envs, seed=seed)


def rollout_throughput(n_envs, budget, warmup):
    """env-steps/sec of the collection loop: policy predict + vec.step()."""
    vec = make_vec(n_envs)
    obs = vec.reset()
    acts, _ = ppo.policy.predict(obs, deterministic=False)
    steps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < warmup:
        acts, _ = ppo.policy.predict(obs, deterministic=False)
        obs, _rew, _dones, _infos = vec.step(acts)
        steps += n_envs
    t1 = time.perf_counter()
    s1 = steps
    while time.perf_counter() - t1 < budget:
        acts, _ = ppo.policy.predict(obs, deterministic=False)
        obs, _rew, _dones, _infos = vec.step(acts)
        steps += n_envs
    t2 = time.perf_counter()
    vec.close()
    return (steps - s1) / (t2 - t1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=None, help='test a single n_envs')
    ap.add_argument('--budget', type=float, default=20.0, help='measure window (s)')
    ap.add_argument('--warmup', type=float, default=6.0)
    ap.add_argument('--trials', type=int, default=1,
                    help='repeat each K and report the best throughput '
                         '(averages out competing-load jitter)')
    args = ap.parse_args()

    global ppo
    ppo = PPO.load(_MODEL, device='cpu')

    if args.k is not None:
        ks = [args.k]
    else:
        ks = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24]

    print(f'n_envs | env-steps/s (best) | per-env step/s | policy predict ms')
    for k in ks:
        best = 0.0
        for _ in range(args.trials):
            best = max(best, rollout_throughput(k, args.budget, args.warmup))
        # batch-predict latency for this K (dominates when K is large)
        obs = {
            'grid': np.zeros((k, 32, 18, 15), dtype=np.float32),
            'hand': np.zeros((k, 5), dtype=np.int32),
            'elixir': np.zeros((k, 1), dtype=np.float32),
        }
        for _ in range(3):
            ppo.policy.predict(obs, deterministic=False)
        t0 = time.perf_counter()
        for _ in range(20):
            ppo.policy.predict(obs, deterministic=False)
        pred_ms = 1e3 * (time.perf_counter() - t0) / 20
        print(f'   {k:>3} | {best:20.0f} | {best / k:14.1f} | {pred_ms:14.1f}')


if __name__ == '__main__':
    main()

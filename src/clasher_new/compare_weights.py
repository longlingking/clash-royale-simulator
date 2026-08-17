"""Compare the best checkpoint against the newest training checkpoint, each
vs a random opponent, and report who wins more.

The "best" weight (`best_model.zip`) is what `BestWeightCallback` selected
against a FIXED eval crowd and is the one you should deploy. The "latest"
weight is the most recent self-play snapshot from `cr_logs/`. In self-play the
latest snapshot is NOT the best one (the opponent set drifts, so winrate vs
current-self is meaningless), so this script quantifies the gap directly
against a fixed random baseline.

Both models play the same role (player 0) with greedy/deterministic actions,
so the only randomness is the random opponent and the per-game deck shuffle.

Parallelism: as in training (train.py / benchmark_env.py), each model's games
run across ``n_envs`` subprocess envs. The moment an env finishes a game it is
reset and immediately starts the next, so all cores stay busy until exactly
``n_games`` completions are counted (in-flight games are discarded). `best`
runs first to completion, then `latest` — so the two models are never compared
under different machine load.

Usage:
    python compare_weights.py [--n-games 50] [--n-envs 8] [--best best_model.zip]
                              [--latest cr_logs/cr_XXX_steps.zip]
"""
import argparse
import glob
import os
import re

import numpy as np
import torch

# card_utils.py opens gamedata.json relative to CWD, so pin the working
# directory to this script's location before importing anything game-related.
_BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_BASE)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from environment import CREnv, random_strategy


def _latest_checkpoint(log_dir='cr_logs', prefix='cr'):
    """Newest cr_logs/cr_*_steps.zip by parsed step count.

    The filenames are not zero-padded to a fixed width, so a lexicographic max
    (e.g. `sorted(...)[-1]`) would return cr_90000 instead of cr_470000.
    """
    pattern = os.path.join(log_dir, f'{prefix}_*_steps.zip')
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f'no checkpoints match {pattern}')
    def step(path):
        m = re.search(rf'{prefix}_(\d+)_steps', os.path.basename(path))
        return int(m.group(1))
    return max(matches, key=step)


class _WinnerInfo(gym.Wrapper):
    """Re-export ``info['winner']`` on the terminal step.

    ``CREnv.step`` returns the game-over flag but not the winner, and the
    winner lives inside the subprocess env. SubprocVecEnv pickles ``info``
    back to the main process, so this wrapper is how play_games counts wins.
    """

    def step(self, action):
        obs, reward, termination, truncation, info = self.env.step(action)
        if termination or truncation:
            info = dict(info)
            info['winner'] = self.unwrapped.battle.winner
        return obs, reward, termination, truncation, info


def _place_obs(stacked, single, i):
    """Overwrite slot ``i`` of a stacked dict observation with one env's."""
    for key, arr in single.items():
        stacked[key][i] = arr
    return stacked


def play_games(model, n_games, n_envs=8):
    """Play ``n_games`` as player 0 vs a random opponent, in parallel.

    Each of the ``n_envs`` subprocess envs plays its own game; a finished env
    is reset via ``vec.env_method`` and immediately starts the next. Counting
    stops at exactly ``n_games`` completed games (in-flight games are
    discarded). Returns the number of wins.
    """
    def make_env(_rank):
        def _init():
            # Subprocess envs run the sim + opponent inference themselves; keep
            # them single-threaded so the processes don't fight over torch's pool.
            torch.set_num_threads(1)
            env = CREnv(opponent_model=None)
            env.opponent = random_strategy
            return _WinnerInfo(env)
        return _init

    vec = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    obs = vec.reset()
    wins = 0
    completed = 0
    last_printed = 0
    try:
        while completed < n_games:
            actions, _ = model.predict(obs, deterministic=True)
            obs, _reward, dones, infos = vec.step(actions)
            for i in np.where(dones)[0]:
                completed += 1
                wins += int(infos[i]['winner'] == 0)
                if completed < n_games:  # start a fresh game in this slot
                    reset_obs = vec.env_method('reset', indices=[i])[0]
                    _place_obs(obs, reset_obs[0], int(i))
            if completed >= last_printed + 10:
                last_printed = completed
                print(f'    game {completed}/{n_games} (wins so far: {wins})')
    finally:
        vec.close()  # reap the subprocess envs (avoids a hang on exit)
    return wins


def main():
    ap = argparse.ArgumentParser(
        description='Compare best vs latest checkpoint against a random opponent.')
    ap.add_argument('--n-games', type=int, default=50,
                    help='games each weight plays vs random (default 50)')
    ap.add_argument('--n-envs', type=int, default=8,
                    help='parallel subprocess envs (default 8; see benchmark_env.py)')
    ap.add_argument('--best', default='best_model.zip')
    ap.add_argument('--latest', default=None,
                    help='explicit checkpoint path; default: newest in cr_logs/')
    args = ap.parse_args()

    latest = args.latest or _latest_checkpoint()

    print(f'loading best   -> {args.best}')
    best_model = PPO.load(args.best, device='cpu')
    print(f'loading latest -> {latest}')
    latest_model = PPO.load(latest, device='cpu')

    n = args.n_games
    print(f'\nbest   ({os.path.basename(args.best)}) vs random, {n} games, {args.n_envs} parallel envs')
    best_wins = play_games(best_model, n, n_envs=args.n_envs)
    print(f'\nlatest ({os.path.basename(latest)}) vs random, {n} games, {args.n_envs} parallel envs')
    latest_wins = play_games(latest_model, n, n_envs=args.n_envs)

    bw, lw = best_wins, latest_wins
    bp = 100.0 * bw / n
    lp = 100.0 * lw / n
    diff_w, diff_p = bw - lw, bp - lp
    # ~1-sigma standard error of the difference of two independent winrates.
    se = 100.0 * ((bp / 100 * (1 - bp / 100) + lp / 100 * (1 - lp / 100)) / n) ** 0.5

    print('\n' + '=' * 60)
    print(f'{"weight":<10}{"wins":>10}{"winrate":>12}')
    print(f'{"best":<10}{bw:>4}/{n}{bp:>9.1f}%')
    print(f'{"latest":<10}{lw:>4}/{n}{lp:>9.1f}%')
    print('-' * 60)
    print(f'difference (best - latest): {diff_w:+d} wins ({diff_p:+.1f} pp)')
    print(f'~1-sigma error of that difference: +/-{se:.1f} pp')
    print('=' * 60)
    if abs(diff_p) < se:
        print(f'verdict: {abs(diff_w)} wins apart is within ~1 sigma noise '
              f'({se:.1f} pp) - treat as roughly even')
    elif diff_p > 0:
        print(f'verdict: best wins more ({diff_w:+d} wins)')
    else:
        print(f'verdict: latest wins more ({diff_w:+d} wins)')


if __name__ == '__main__':
    # Single-thread inference: faster and quieter than 16 cores fighting over
    # a 0.77M-param forward pass.
    torch.set_num_threads(1)
    main()

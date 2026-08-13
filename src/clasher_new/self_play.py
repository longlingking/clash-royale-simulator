"""Self-play plumbing for the Clash Royale simulator.

What this module adds on top of ``train.py``:

- fixed *scripted* opponents (baselines that never learn);
- ``OpponentPool``: picks the training opponent from a weighted mix of
  recent self-play checkpoints, base checkpoints and fixed strategies;
- ``evaluate_agent``: winrate of a model against a fixed crowd;
- ``OpponentSwapCallback`` / ``BestWeightCallback``: the two SB3 callbacks
  that wire the pool and the best-model selection into ``model.learn()``.
"""
import glob
import os
import random

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from environment import CREnv, random_strategy, entity_names
from card_utils import Card


# ---------------------------------------------------------------------------
# Fixed scripted opponents
#
# A script returns an action tuple ``(slot, y, x)`` in player-1's frame,
# exactly like the RL agent's action space. CREnv maps it to the arena with:
#   abs_x = 18 - (x + 0.5),  abs_y = 32 - (y + 0.5)
# so a small y_action / x_action lands on player 1's own side.
# ---------------------------------------------------------------------------

_BRIDGE_RIGHT = (13, 3)   # -> absolute (14.5, 18.5), just below the right bridge
_BRIDGE_LEFT = (13, 14)   # -> absolute ( 3.5, 18.5), just below the left bridge
_KING_AREA = (5, 8)       # -> absolute ( 9.5, 26.5), near the king tower

_CARD_COST = {}


def _card_cost(entity_id):
    if entity_id not in _CARD_COST:
        _CARD_COST[entity_id] = Card(entity_names[entity_id]).elixir
    return _CARD_COST[entity_id]


def _affordable_slot(obs):
    """Cheapest playable card among the 4 in hand, as slot 1..4 (0 = none)."""
    hand = obs['hand']
    elixir = float(obs['elixir'][0])
    best_slot, best_cost = 0, float('inf')
    for i, entity_id in enumerate(hand[:4]):
        if entity_id == 0:  # 'None' padding, shouldn't happen
            continue
        cost = _card_cost(int(entity_id))
        if cost <= elixir and cost < best_cost:
            best_slot, best_cost = i + 1, cost
    return best_slot


def bridge_rush_script(obs):
    """Always push the cheapest affordable card toward the right bridge."""
    slot = _affordable_slot(obs)
    if slot == 0:
        return (0, 0, 0)
    y, x = _BRIDGE_RIGHT
    return (slot, y, x)


def bridge_rush_left_script(obs):
    """Always push the cheapest affordable card toward the left bridge."""
    slot = _affordable_slot(obs)
    if slot == 0:
        return (0, 0, 0)
    y, x = _BRIDGE_LEFT
    return (slot, y, x)


def defender_script(obs):
    """Park the cheapest affordable card near the king tower."""
    slot = _affordable_slot(obs)
    if slot == 0:
        return (0, 0, 0)
    y, x = _KING_AREA
    return (slot, y, x)


# ---------------------------------------------------------------------------
# Opponent helpers
# ---------------------------------------------------------------------------

def _model_opponent(path, device='cpu', deterministic=True):
    """Turn a checkpoint path into an ``obs -> action`` callable for player 1."""
    model = PPO.load(path, device=device)
    return lambda obs: model.predict(obs, deterministic=deterministic)[0]


# ---------------------------------------------------------------------------
# OpponentPool
# ---------------------------------------------------------------------------

class OpponentPool:
    """Weighted mix of opponent candidates.

    Each ``pick()`` returns a callable ``obs -> (slot, y, x)`` that CREnv uses
    as the player-1 opponent for the next training window. The mix keeps the
    agent honest: most of the time it fights recent versions of itself, but it
    also meets old snapshots and dumb fixed strategies so it doesn't overfit
    to a single opponent.

    Parameters
    ----------
    base_checkpoints : list[str]
        Checkpoints that already exist on disk (frozen during this run).
    recent_dir : str
        Directory written by SB3's CheckpointCallback.
    prefix : str
        CheckpointCallback's name_prefix.
    n_recent : int
        How many of the most recent self-play checkpoints are eligible.
    fixed_strategies : list[callable]
        Non-learning opponents (scripts / random) that never change.
    weights : dict
        Relative weights for the three categories: recent / base / fixed.
    deterministic : bool
        Whether loaded models play greedily or sample actions.
    """

    def __init__(self, base_checkpoints=(), recent_dir='cr_logs', prefix='cr',
                 n_recent=6, fixed_strategies=None, weights=None,
                 deterministic=False, device='cpu', seed=None):
        self.base_checkpoints = list(base_checkpoints)
        self.recent_dir = recent_dir
        self.prefix = prefix
        self.n_recent = n_recent
        self.fixed_strategies = list(fixed_strategies) if fixed_strategies else [random_strategy]
        self.deterministic = deterministic
        self.device = device
        self.weights = dict(weights or {'recent': 0.6, 'base': 0.2, 'fixed': 0.2})
        self._rng = random.Random(seed)
        self._cache = {}  # path -> loaded PPO (bounds memory)

    def _recent_paths(self):
        pattern = os.path.join(self.recent_dir, f'{self.prefix}_*_steps.zip')
        return sorted(glob.glob(pattern))[-self.n_recent:]

    def _load(self, path):
        if path not in self._cache:
            self._cache[path] = PPO.load(path, device=self.device)
            while len(self._cache) > self.n_recent + len(self.base_checkpoints) + 2:
                del self._cache[next(iter(self._cache))]
        return self._cache[path]

    def _model_opponent(self, path):
        model = self._load(path)
        return lambda obs: model.predict(obs, deterministic=self.deterministic)[0]

    def pick(self):
        """Return one opponent callable for the next training window."""
        choices = []  # (weight, 'path'|'fn', payload)
        recent = self._recent_paths()
        if recent:
            choices.append((self.weights['recent'], 'path', self._rng.choice(recent)))
        if self.base_checkpoints:
            choices.append((self.weights['base'], 'path', self._rng.choice(self.base_checkpoints)))
        if self.fixed_strategies:
            choices.append((self.weights['fixed'], 'fn', self._rng.choice(self.fixed_strategies)))
        if not choices:
            return random_strategy

        r = self._rng.random() * sum(w for w, _, _ in choices)
        for w, kind, payload in choices:
            r -= w
            if r <= 0:
                return self._model_opponent(payload) if kind == 'path' else payload
        kind, payload = choices[-1][1], choices[-1][2]
        return self._model_opponent(payload) if kind == 'path' else payload


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(agent, opponents, n_games=20, agent_deterministic=True, progress_every=10):
    """Winrate of ``agent`` (player 0) against each opponent (player 1).

    ``opponents`` is a dict ``name -> callable`` (or a checkpoint path, which
    is loaded here). Returns ``{name: winrate}``. The agent plays greedily so
    the numbers are a stable measure of policy quality. Prints progress every
    ``progress_every`` games (pass 0 to silence).
    """
    results = {}
    env = CREnv(opponent_model=None)
    for name, opp in opponents.items():
        if isinstance(opp, str):
            opp = _model_opponent(opp)
        env.opponent = opp
        wins = 0
        for i in range(n_games):
            if progress_every and i % progress_every == 0:
                print(f'  [{name}] game {i}/{n_games}')
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = agent.predict(obs, deterministic=agent_deterministic)
                obs, _, termination, truncation, _ = env.step(action)
                done = termination or truncation
            wins += int(env.battle.winner == 0)
        results[name] = wins / n_games
        print(f'  [{name}] {wins}/{n_games} wins -> {results[name]:.2f}')
    return results


# ---------------------------------------------------------------------------
# SB3 callbacks
# ---------------------------------------------------------------------------

class OpponentSwapCallback(BaseCallback):
    """Re-pick the training opponent from the pool every ``swap_every`` steps."""

    def __init__(self, env, pool, swap_every=2048, verbose=0):
        super().__init__(verbose)
        self.env = env  # the raw CREnv used for training (not the VecEnv wrapper)
        self.pool = pool
        self.swap_every = swap_every

    def _init_callback(self):
        self.env.opponent = self.pool.pick()

    def _on_step(self):
        if self.n_calls % self.swap_every == 0:
            self.env.opponent = self.pool.pick()
        return True


class BestWeightCallback(BaseCallback):
    """Every ``eval_every`` steps, measure winrate vs a FIXED crowd.

    The crowd never changes during a run (that's the whole point: it's the
    stable ruler used to select the best checkpoint). When the score improves,
    the current weights are saved to ``best_path``.

    Parameters
    ----------
    opponents : dict
        ``name -> callable`` or checkpoint path. Fixed for the whole run.
    eval_every : int
        Absolute timestep cadence for evaluation (works with continuation).
    n_games : int
        Games per opponent per evaluation.
    best_path : str
        Where the best-so-far weights are written.
    eval_at_start : bool
        Evaluate the loaded model once before training starts, seeding
        ``best_path`` and the comparison baseline.
    """

    def __init__(self, opponents, eval_every=100_000, n_games=20,
                 best_path='best_model.zip', eval_at_start=True, verbose=0):
        super().__init__(verbose)
        self.opponents = dict(opponents)
        self.eval_every = eval_every
        self.n_games = n_games
        self.best_path = best_path
        self.eval_at_start = eval_at_start
        self.best_score = -np.inf
        self._last_eval_step = -1
        self._cache = {}  # checkpoint path -> callable, reused across evals

    def _resolve_opponents(self):
        resolved = {}
        for name, opp in self.opponents.items():
            if isinstance(opp, str):
                if opp not in self._cache:
                    self._cache[opp] = _model_opponent(opp)
                resolved[name] = self._cache[opp]
            else:
                resolved[name] = opp
        return resolved

    def _init_callback(self):
        self._last_eval_step = -1
        if self.eval_at_start:
            self._evaluate()

    def _on_step(self):
        if (self.num_timesteps > 0
                and self.num_timesteps != self._last_eval_step
                and self.num_timesteps % self.eval_every == 0):
            self._evaluate()
        return True

    def _evaluate(self):
        self._last_eval_step = self.num_timesteps
        results = evaluate_agent(self.model, self._resolve_opponents(), self.n_games)
        score = float(np.mean(list(results.values())))
        self.logger.record('eval/mean_winrate', score)
        for name, wr in results.items():
            self.logger.record(f'eval/winrate_{name}', wr)
        self.logger.record('eval/best_score', self.best_score)
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.best_path)
            self.logger.record('eval/best_score', self.best_score)
            if self.verbose:
                print(f'[best] new best {score:.3f} -> saved {self.best_path}')
        return score


if __name__ == '__main__':
    # Standalone check: winrate of the loaded model vs each fixed opponent.
    # Usage: `python self_play.py [n_games]`  (default 50 — with 2 games the
    # winrates are pure noise).
    import sys
    import torch
    # Tiny model: single-thread inference is faster (and quieter) than having
    # every core fight over a 0.62M-param forward pass.
    torch.set_num_threads(1)
    _base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(_base)  # card_utils opens gamedata.json relative to CWD
    _model = PPO.load(os.path.join(_base, 'cr_discrete'))
    _n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    _crowd = {
        'random': random_strategy,
        'bridge_rush': bridge_rush_script,
        'defender': defender_script,
        'cr_discrete': os.path.join(_base, 'cr_discrete.zip'),
    }
    print(evaluate_agent(_model, _crowd, n_games=_n))

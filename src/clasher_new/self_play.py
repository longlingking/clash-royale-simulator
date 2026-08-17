"""Self-play plumbing for the Clash Royale simulator.

What this module adds on top of ``train.py``:

- fixed *scripted* opponents (baselines that never learn);
- ``OpponentPool``: picks the training opponent from a weighted mix of
  recent self-play checkpoints, base checkpoints and fixed strategies, with
  optional per-candidate *adaptive* priorities (future.md #2);
- ``OpponentEpisodeWrapper`` / ``make_opponent_vec_env``: per-episode opponent
  sampling inside a ``SubprocVecEnv`` (IsaacLab-style batch-level mixing);
- ``evaluate_agent``: winrate of a model against a fixed crowd;
- ``OpponentSwapCallback`` (legacy window-level swapping) / ``BestWeightCallback``
  / ``AdaptiveWeightCallback``: the SB3 callbacks that wire the pool, the
  best-model selection and the adaptive opponent weights into ``model.learn()``.
"""
import glob
import os
import random

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

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
                 deterministic=False, device='cpu', seed=None, priorities=None):
        self.base_checkpoints = list(base_checkpoints)
        self.recent_dir = recent_dir
        self.prefix = prefix
        self.n_recent = n_recent
        self.fixed_strategies = list(fixed_strategies) if fixed_strategies else [random_strategy]
        self.deterministic = deterministic
        self.device = device
        self.weights = dict(weights or {'recent': 0.6, 'base': 0.2, 'fixed': 0.2})
        self.priorities = dict(priorities or {})  # candidate key -> sampling weight
        self.last_key = None  # key of the opponent most recently handed out
        self._rng = random.Random(seed)
        self._cache = {}  # path -> loaded PPO (bounds memory)

    def _recent_paths(self):
        pattern = os.path.join(self.recent_dir, f'{self.prefix}_*_steps.zip')
        # Sort by the step count embedded in the filename, NOT by the
        # filename string: lexicographic order breaks once step counts cross a
        # digit-length boundary ('cr_100000_steps.zip' < 'cr_20000_steps.zip').
        suffix = '_steps.zip'
        def step(path):
            base = os.path.basename(path)
            return int(base[len(self.prefix) + 1:-len(suffix)])
        return sorted(glob.glob(pattern), key=step)[-self.n_recent:]

    def _load(self, path):
        if path not in self._cache:
            self._cache[path] = PPO.load(path, device=self.device)
            while len(self._cache) > self.n_recent + len(self.base_checkpoints) + 2:
                del self._cache[next(iter(self._cache))]
        return self._cache[path]

    def _model_opponent(self, path):
        model = self._load(path)
        return lambda obs: model.predict(obs, deterministic=self.deterministic)[0]

    def child(self, rank, base_seed=None):
        """A fresh pool for one vectorised-env subprocess.

        Each child of a ``SubprocVecEnv`` needs its own RNG — otherwise every
        env draws the same opponent in lock-step instead of mixing — and its
        own model cache, so checkpoints load lazily inside the child instead
        of being pickled across processes.
        """
        return OpponentPool(
            base_checkpoints=self.base_checkpoints,
            recent_dir=self.recent_dir,
            prefix=self.prefix,
            n_recent=self.n_recent,
            fixed_strategies=self.fixed_strategies,
            weights=self.weights,
            priorities=self.priorities,
            deterministic=self.deterministic,
            device=self.device,
            seed=(base_seed + rank) if base_seed is not None else None,
        )

    # ------------------------------------------------------------------
    # Candidate keys + adaptive priorities (future.md #2)
    #
    # Every candidate has a stable key: 'recent:cr_123456_steps.zip',
    # 'base:cr_discrete.zip', 'fixed:bridge_rush_script'. priorities[key]
    # scales its sampling share; the main-process AdaptiveWeightCallback
    # derives these from in-training winrates and pushes them here. With no
    # priorities set (all default 1.0) sampling is bit-for-bit the old
    # uniform two-stage scheme.
    # ------------------------------------------------------------------

    @staticmethod
    def _key(bucket, payload):
        if bucket == 'fixed':
            return f'fixed:{payload.__name__}'
        return f'{bucket}:{os.path.basename(payload)}'

    def set_priorities(self, priorities):
        """Replace the per-candidate sampling priorities (full dict)."""
        self.priorities = dict(priorities)

    def _priority(self, key):
        return self.priorities.get(key, 1.0)

    def _weighted_candidates(self):
        """Flat ``[(weight, kind, payload, key)]`` for one weighted draw.

        ``weight = base_bucket_weight / bucket_size * priority``, so within a
        bucket members are drawn in proportion to their priority, and a
        bucket's total share scales with the mean priority of its members
        (the whole fixed bucket shrinks once every script is crushed).
        """
        entries = []  # (weight, kind, payload, key)
        recent = self._recent_paths()
        if recent:
            w = self.weights['recent'] / len(recent)
            for path in recent:
                key = self._key('recent', path)
                entries.append((w * self._priority(key), 'path', path, key))
        if self.base_checkpoints:
            w = self.weights['base'] / len(self.base_checkpoints)
            for path in self.base_checkpoints:
                key = self._key('base', path)
                entries.append((w * self._priority(key), 'path', path, key))
        if self.fixed_strategies:
            w = self.weights['fixed'] / len(self.fixed_strategies)
            for fn in self.fixed_strategies:
                key = self._key('fixed', fn)
                entries.append((w * self._priority(key), 'fn', fn, key))
        return entries

    def pick(self):
        """Return one opponent callable for the next training episode.

        Weighted draw over all current candidates (bucket base weight times the
        candidate's adaptive priority). Also records ``self.last_key`` so
        callers can attribute the episode result to the right opponent.
        """
        entries = self._weighted_candidates()
        if not entries:
            self.last_key = 'fixed:random_strategy'
            return random_strategy

        total = sum(w for w, _, _, _ in entries)
        r = self._rng.random() * total
        for w, kind, payload, key in entries:
            r -= w
            if r <= 0:
                self.last_key = key
                return self._model_opponent(payload) if kind == 'path' else payload
        _, kind, payload, key = entries[-1]
        self.last_key = key
        return self._model_opponent(payload) if kind == 'path' else payload


# ---------------------------------------------------------------------------
# Per-episode opponent sampling for vectorised training (future.md #1)
# ---------------------------------------------------------------------------

class OpponentEpisodeWrapper(gym.Wrapper):
    """Re-pick the training opponent from a pool at every ``reset()``.

    When used inside a ``SubprocVecEnv`` this replaces window-level swapping
    (``OpponentSwapCallback``): each sub-env re-draws its opponent per
    episode, so a single PPO rollout mixes trajectories from all opponent
    types in proportion to the pool weights (IsaacLab-style batch mixing).

    It also feeds :class:`AdaptiveWeightCallback` (future.md #2): on every
    finished episode it reports which opponent was played and whether player 0
    (the agent) won, via ``info['opponent']`` / ``info['won']``. ``Monitor``
    (outermost) preserves these keys, so they reach the main process.
    """

    def __init__(self, env, pool):
        super().__init__(env)
        self.pool = pool
        self._current_key = None

    def reset(self, **kwargs):
        self.unwrapped.opponent = self.pool.pick()
        # The pool records which opponent it just handed out; remember it so we
        # can attribute this episode's result when it ends.
        self._current_key = getattr(self.pool, 'last_key', None)
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, termination, truncation, info = self.env.step(action)
        if (termination or truncation) and self._current_key is not None:
            info = dict(info)
            info['opponent'] = self._current_key
            info['won'] = int(self.unwrapped.battle.winner == 0)
        return obs, reward, termination, truncation, info

    def set_priorities(self, priorities):
        """Forward adaptive priorities from the main process to the pool."""
        if hasattr(self.pool, 'set_priorities'):
            self.pool.set_priorities(priorities)

    def get_priorities(self):
        """Read the pool's priorities back (used by tests / debugging)."""
        return dict(getattr(self.pool, 'priorities', {}))


def make_opponent_vec_env(pool, n_envs=8, seed=None, visualize=False, speed=1.0,
                          start_method=None):
    """A ``SubprocVecEnv`` of ``n_envs`` CREnv for self-play training.

    Each sub-env is wrapped in :class:`OpponentEpisodeWrapper` and holds its
    own ``pool.child(rank)``, so every episode draws its opponent from the
    pool independently. Model checkpoints load lazily inside each child (never
    pickled across processes) and the deck lists are per-process copies, so
    envs don't stomp on each other's shuffle state.

    The envs are also wrapped in ``Monitor`` so SB3's PPO can log
    ``rollout/ep_rew_mean`` / ``ep_len_mean`` (it reads episode rewards from
    ``info["episode"]``, which only ``Monitor`` injects).
    """
    def make_env(rank):
        def _init():
            # Children run their own opponent inference; keep them
            # single-threaded so K subprocesses don't fight over torch's pool.
            torch.set_num_threads(1)
            env = CREnv(opponent_model=None, visualize=visualize, speed=speed)
            env = OpponentEpisodeWrapper(env, pool.child(rank, base_seed=seed))
            # filename=None: no CSV, we only need info["episode"] for logging.
            return Monitor(env, filename=None)
        return _init

    return SubprocVecEnv([make_env(i) for i in range(n_envs)],
                         start_method=start_method)


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


def adapt_priorities(buffer, old, alpha=0.3, floor=0.1):
    """Turn a ``{key: (wins, games)}`` buffer into smoothed priorities.

    ``priority = max(floor, 1 - winrate)`` — opponents that still beat the
    agent get a high priority (worth sampling more), ones it crushes fall to
    the floor (sampled rarely but never zeroed, so the agent can't forget
    them). The result is exponentially smoothed against ``old`` so weights
    don't chase single-episode noise. Pure function: unit-testable without an
    env.
    """
    out = {}
    for key, (wins, games) in buffer.items():
        if games <= 0:
            continue
        target = max(floor, 1.0 - wins / games)
        out[key] = alpha * target + (1.0 - alpha) * old.get(key, 1.0)
    return out


class AdaptiveWeightCallback(BaseCallback):
    """Re-sample the opponent pool from in-training winrates (future.md #2).

    Every finished training episode reports ``(opponent, won)`` through
    ``info`` (see :class:`OpponentEpisodeWrapper`). This callback accumulates
    those in the main process and, every ``update_every`` env-steps, converts
    the per-candidate winrate into a sampling priority (via
    :func:`adapt_priorities`) and pushes the new priorities into every
    sub-env's ``OpponentPool`` through ``env_method``. ``pick()`` then spends
    more episodes on opponents that still beat the agent and fewer on ones it
    crushes.

    The FIXED eval crowd used by :class:`BestWeightCallback` is untouched —
    that stays the stable ruler for best-model selection.
    """

    def __init__(self, update_every=50_000, alpha=0.3, floor=0.1, verbose=0):
        super().__init__(verbose)
        self.update_every = update_every
        self.alpha = alpha
        self.floor = floor
        self._buffer = {}      # candidate key -> [wins, games] since last update
        self._priorities = {}  # candidate key -> smoothed priority
        self._last_update = -1

    @staticmethod
    def _tag(key):
        # ':' and '.' are awkward in tensorboard metric names.
        return key.replace(':', '_').replace('.', '_').replace('/', '_')

    def _init_callback(self):
        self._last_update = -1

    def _on_step(self):
        # One call per vec-step; infos/dones are the per-env lists.
        for info, done in zip(self.locals.get('infos', []),
                              self.locals.get('dones', [])):
            if done and 'opponent' in info and 'won' in info:
                key = info['opponent']
                wins, games = self._buffer.get(key, (0, 0))
                self._buffer[key] = (wins + int(info['won']), games + 1)
        if (self.num_timesteps > 0
                and self.num_timesteps != self._last_update
                and self.num_timesteps % self.update_every == 0):
            self._update()
        return True

    def _update(self):
        self._last_update = self.num_timesteps
        updated = adapt_priorities(self._buffer, self._priorities,
                                   alpha=self.alpha, floor=self.floor)
        if self.logger is not None:
            for key, (wins, games) in self._buffer.items():
                if games > 0:
                    tag = self._tag(key)
                    self.logger.record(f'adap/winrate_{tag}', wins / games)
                    self.logger.record(f'adap/priority_{tag}', updated[key])
        self._priorities.update(updated)
        self._buffer = {}
        if not updated:
            return
        env = self.training_env
        if env is not None and hasattr(env, 'env_method'):
            env.env_method('set_priorities', self._priorities)
        if self.verbose:
            short = {k.split(':')[-1]: v for k, v in sorted(updated.items())}
            print('[adap] priorities: ' + ', '.join(f'{k}={v:.2f}' for k, v in short.items()))


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

"""Tests for per-env opponent sampling (future.md #1).

The goal: replace *window-level* opponent swapping (``OpponentSwapCallback``
switches every 2048 steps) with *per-episode* sampling inside a vectorised
environment, so each PPO rollout batch naturally mixes trajectories from all
opponent types in proportion to the pool weights.

These tests pin down four pieces:

1. ``OpponentEpisodeWrapper`` — re-picks the opponent at every ``reset()``.
2. ``OpponentPool.child`` — gives each vectorised sub-env an independent pool
   (own RNG, empty model cache), so envs don't draw opponents in lock-step.
3. ``make_opponent_vec_env`` — builds a ``SubprocVecEnv`` that actually rolls
   out against freshly-picked opponents, and reports episode rewards.
4. ``OpponentPool._recent_paths`` — the "recent self-play checkpoints" bucket
   must pick the newest by *step count*, not by filename string order.
"""
import os

import numpy as np
import pytest

from stable_baselines3.common.vec_env import SubprocVecEnv

from environment import CREnv, random_strategy
from self_play import (
    OpponentPool,
    OpponentEpisodeWrapper,
    make_opponent_vec_env,
    bridge_rush_left_script,
    bridge_rush_script,
    defender_script,
)


class StubPool:
    """A stand-in pool: hands out opponents from a fixed sequence."""

    def __init__(self, opponents):
        self._opponents = list(opponents)
        self._i = 0
        self.picks = []

    def pick(self):
        opp = self._opponents[self._i % len(self._opponents)]
        self._i += 1
        self.picks.append(opp)
        return opp


def test_recent_paths_selects_newest_by_step_not_lexicographic(tmp_path):
    # Regression: the old code did ``sorted(glob(...))``, which orders
    # filenames as strings. Once step counts cross a digit-length boundary
    # ('100000' < '20000' lexicographically) the "most recent" opponents were
    # the wrong checkpoints — recent self-play weights would be stale/missing.
    for step in [10000, 20000, 30000, 100000, 110000, 120000]:
        (tmp_path / f'cr_{step}_steps.zip').touch()

    pool = OpponentPool(recent_dir=str(tmp_path), prefix='cr', n_recent=3,
                        fixed_strategies=[], seed=0)
    steps = [int(os.path.basename(p).split('_')[1]) for p in pool._recent_paths()]
    assert steps == [100000, 110000, 120000]


# ---------------------------------------------------------------------------
# OpponentEpisodeWrapper
# ---------------------------------------------------------------------------

def test_wrapper_re_picks_opponent_on_every_reset():
    opps = [random_strategy, bridge_rush_script, defender_script]
    pool = StubPool(opps)
    wrapped = OpponentEpisodeWrapper(CREnv(opponent_model=None), pool)

    wrapped.reset()
    assert wrapped.unwrapped.opponent is opps[0]

    wrapped.reset()
    assert wrapped.unwrapped.opponent is opps[1]

    assert pool.picks == opps[:2]


def test_wrapper_steps_against_the_picked_opponent():
    # The picked opponent must actually be the one the env plays against.
    pool = StubPool([bridge_rush_script])
    wrapped = OpponentEpisodeWrapper(CREnv(opponent_model=None), pool)
    wrapped.reset()

    obs, reward, terminated, truncated, info = wrapped.step((0, 0, 0))
    assert obs['grid'].shape == (32, 18, 15)
    # After one step we can't tell the opponent apart, but the game advanced.
    assert wrapped.unwrapped.battle is not None


def test_wrapper_reset_passes_seed_and_options_through():
    pool = StubPool([bridge_rush_script])
    wrapped = OpponentEpisodeWrapper(CREnv(opponent_model=None), pool)
    obs, info = wrapped.reset(seed=1234)
    assert obs['grid'].shape == (32, 18, 15)
    assert info == {}


# ---------------------------------------------------------------------------
# OpponentPool.child
# ---------------------------------------------------------------------------

def test_pool_child_keeps_config_but_gets_its_own_rng_and_cache():
    pool = OpponentPool(
        base_checkpoints=['/tmp/some_checkpoint.zip'],
        recent_dir='/tmp/does_not_exist',
        prefix='cr',
        n_recent=2,
        fixed_strategies=[random_strategy, bridge_rush_script,
                          bridge_rush_left_script, defender_script],
        weights={'recent': 0.6, 'base': 0.2, 'fixed': 0.2},
        seed=0,
    )

    child = pool.child(rank=0, base_seed=0)

    # Same configuration ...
    assert child.base_checkpoints == pool.base_checkpoints
    assert child.fixed_strategies == pool.fixed_strategies
    assert child.weights == pool.weights
    assert child.recent_dir == pool.recent_dir
    # ... but a separate object with its own (empty) model cache.
    assert child is not pool
    assert child._cache == {}


def test_pool_child_ranks_draw_different_opponents(tmp_path):
    # Different ranks must produce different opponent sequences (independent
    # RNG), otherwise all sub-envs would face the same opponent in lock-step.
    # recent_dir must be empty so the "recent" bucket stays inactive (it would
    # otherwise pick up real checkpoints from cr_logs and return model opps).
    pool = OpponentPool(
        recent_dir=str(tmp_path),
        fixed_strategies=[random_strategy, bridge_rush_script,
                          bridge_rush_left_script, defender_script],
        seed=0,
    )
    c0 = pool.child(0, base_seed=0)
    c1 = pool.child(1, base_seed=0)
    seq0 = [c0.pick() for _ in range(8)]
    seq1 = [c1.pick() for _ in range(8)]
    assert seq0 != seq1
    assert set(seq0) <= set(pool.fixed_strategies)
    assert set(seq1) <= set(pool.fixed_strategies)


# ---------------------------------------------------------------------------
# make_opponent_vec_env
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('start_method', ['fork'])
def test_make_opponent_vec_env_returns_subprocvecenv_of_size(start_method, tmp_path):
    pool = OpponentPool(recent_dir=str(tmp_path),
                        fixed_strategies=[bridge_rush_script], seed=0)
    vec = make_opponent_vec_env(pool, n_envs=2, seed=0, start_method=start_method)
    try:
        assert isinstance(vec, SubprocVecEnv)
        # VecEnv has no ``len`` — ``num_envs`` is the public attribute.
        assert vec.num_envs == 2
    finally:
        vec.close()


@pytest.mark.parametrize('start_method', ['fork'])
def test_make_opponent_vec_env_rollout(start_method, tmp_path):
    pool = OpponentPool(
        recent_dir=str(tmp_path),
        fixed_strategies=[random_strategy, bridge_rush_script], seed=1)
    vec = make_opponent_vec_env(pool, n_envs=2, seed=1, start_method=start_method)
    try:
        obs = vec.reset()
        assert obs['grid'].shape == (2, 32, 18, 15)
        assert obs['hand'].shape == (2, 5)

        for _ in range(10):
            obs, rews, dones, infos = vec.step([[0, 0, 0], [0, 0, 0]])
            assert obs['grid'].shape == (2, 32, 18, 15)
            assert rews.shape == (2,)
            assert dones.shape == (2,)
            # SB3 returns infos as a tuple (one dict per env).
            assert len(infos) == 2
    finally:
        vec.close()


def test_rollout_reports_episode_reward_in_infos(tmp_path):
    # SB3 only logs ``rollout/ep_rew_mean`` when the env is Monitor-wrapped,
    # which injects ``info["episode"]`` at the end of every episode. Step
    # until at least one of the two envs finishes a game and check the signal
    # reaches the parent (this is what feeds the reward line in the logs).
    pool = OpponentPool(recent_dir=str(tmp_path),
                        fixed_strategies=[bridge_rush_script], seed=2)
    vec = make_opponent_vec_env(pool, n_envs=2, seed=2, start_method='fork')
    try:
        obs = vec.reset()
        episode_seen = False
        for _ in range(500):
            obs, rews, dones, infos = vec.step([[0, 0, 0], [0, 0, 0]])
            for info in infos:
                if 'episode' in info:
                    assert 'r' in info['episode'] and 'l' in info['episode']
                    episode_seen = True
            if episode_seen:
                break
        assert episode_seen, "no episode finished within 500 steps"
    finally:
        vec.close()

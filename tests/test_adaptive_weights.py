"""Tests for adaptive opponent weighting (future.md #2).

#2 replaces the static pool weights with *per-candidate priorities* derived
from in-training winrates: opponents that still beat the agent are sampled
more, ones it crushes fall toward a floor. These tests pin down:

1. ``OpponentPool._weighted_candidates`` — the weight formula
   (bucket_base_weight / bucket_size * priority), and that with no priorities
   set it reproduces the old uniform two-stage sampler.
2. ``pick()`` — the actual weighted draw, ``last_key`` tracking.
3. ``adapt_priorities`` — the winrate -> priority conversion and smoothing.
4. ``OpponentEpisodeWrapper`` — reports ``(opponent, won)`` for each finished
   episode so the main process can accumulate results.
5. ``AdaptiveWeightCallback`` — accumulates results and pushes the new
   priorities into every sub-env's pool via ``env_method`` (both a manual
   driver and a real ``model.learn`` smoke).
"""
import pytest

from stable_baselines3 import PPO

from environment import CREnv, random_strategy
from self_play import (
    AdaptiveWeightCallback,
    OpponentPool,
    OpponentEpisodeWrapper,
    adapt_priorities,
    bridge_rush_script,
    defender_script,
    make_opponent_vec_env,
)


class _DummyModel:
    """Minimal stand-in for a PPO so the callback's properties resolve.

    ``BaseCallback.training_env`` and ``.logger`` are properties that read
    from ``self.model``, so tests that drive the callback by hand (without a
    real ``model.learn``) need a tiny stub providing ``get_env()``/``logger``.
    """

    logger = None

    def __init__(self, env):
        self._env = env

    def get_env(self):
        return self._env


def _pool(tmp_path, fixed, weights=None, base=()):
    return OpponentPool(
        recent_dir=str(tmp_path),
        base_checkpoints=list(base),
        fixed_strategies=list(fixed),
        weights=weights,
        seed=0,
    )


# ---------------------------------------------------------------------------
# OpponentPool._weighted_candidates (the weight formula)
# ---------------------------------------------------------------------------

def test_weighted_candidates_default_is_uniform():
    # recent/base empty, fixed bucket base 0.6 over 3 members -> 0.2 each.
    pool = _pool('/tmp/does_not_exist',
                 [random_strategy, defender_script, bridge_rush_script],
                 weights={'recent': 0.4, 'base': 0.0, 'fixed': 0.6})
    weights = [w for w, *_ in pool._weighted_candidates()]
    assert weights == pytest.approx([0.2, 0.2, 0.2])


def test_priorities_weight_within_bucket():
    pool = _pool('/tmp/does_not_exist',
                 [random_strategy, defender_script, bridge_rush_script],
                 weights={'recent': 0.4, 'base': 0.0, 'fixed': 0.6})
    pool.set_priorities({
        'fixed:random_strategy': 0.1,
        'fixed:defender_script': 0.9,
        'fixed:bridge_rush_script': 0.5,
    })
    weights = [w for w, *_ in pool._weighted_candidates()]
    assert weights == pytest.approx([0.6 / 3 * 0.1, 0.6 / 3 * 0.9, 0.6 / 3 * 0.5])


def test_crushed_bucket_shrinks():
    # A base checkpoint crushed to the floor drags its whole bucket down, while
    # untouched scripts keep (almost) their base share.
    pool = _pool('/tmp/does_not_exist',
                 [random_strategy, defender_script],
                 weights={'recent': 0.0, 'base': 0.2, 'fixed': 0.8},
                 base=['/nonexistent/a.zip'])
    pool.set_priorities({
        'base:a.zip': 0.1,
        'fixed:random_strategy': 1.0,
        'fixed:defender_script': 1.0,
    })
    weights = [w for w, *_ in pool._weighted_candidates()]
    total = sum(weights)
    base_share, fixed_share = weights[0] / total, sum(weights[1:]) / total
    # Without adaptation base would be 0.2/(0.2+0.8) = 20%; the floor cuts it
    # to ~2.4%.
    assert base_share < 0.05
    assert fixed_share > 0.95


# ---------------------------------------------------------------------------
# OpponentPool.pick
# ---------------------------------------------------------------------------

def test_pick_distribution_follows_priorities():
    pool = _pool('/tmp/does_not_exist',
                 [random_strategy, defender_script, bridge_rush_script],
                 weights={'recent': 0.0, 'base': 0.0, 'fixed': 1.0})
    pool.set_priorities({
        'fixed:random_strategy': 0.1,
        'fixed:defender_script': 0.9,
        'fixed:bridge_rush_script': 0.5,
    })
    counts = {random_strategy: 0, defender_script: 0, bridge_rush_script: 0}
    for _ in range(6000):
        counts[pool.pick()] += 1
    assert counts[defender_script] > counts[bridge_rush_script] > counts[random_strategy]


def test_pick_records_last_key():
    pool = _pool('/tmp/does_not_exist', [random_strategy, defender_script])
    pool.pick()
    assert pool.last_key in ('fixed:random_strategy', 'fixed:defender_script')


# ---------------------------------------------------------------------------
# adapt_priorities (pure function)
# ---------------------------------------------------------------------------

def test_adapt_priorities_smooths_and_floors():
    out = adapt_priorities(
        {'a': (10, 10), 'b': (0, 10), 'c': (5, 10)},
        {'a': 1.0, 'b': 1.0, 'c': 1.0},
        alpha=0.5, floor=0.2,
    )
    assert out['a'] == pytest.approx(0.6)   # 100% crushed -> floor, smoothed
    assert out['b'] == pytest.approx(1.0)   # never lost -> stay at top
    assert out['c'] == pytest.approx(0.75)  # 50% -> 0.75


# ---------------------------------------------------------------------------
# OpponentEpisodeWrapper: episode-result reporting
# ---------------------------------------------------------------------------

def test_wrapper_reports_episode_result(tmp_path):
    pool = _pool(tmp_path, [bridge_rush_script, defender_script])
    wrapped = OpponentEpisodeWrapper(CREnv(opponent_model=None), pool)
    wrapped.reset()
    assert wrapped._current_key is not None
    done = False
    for _ in range(600):
        obs, rew, term, trunc, info = wrapped.step((0, 0, 0))
        if term or trunc:
            assert info['opponent'] == wrapped._current_key
            assert info['won'] in (0, 1)
            done = True
            break
    assert done, 'no episode finished within 600 steps'


def test_vec_rollout_reports_opponent_and_winner(tmp_path):
    # End-to-end through SubprocVecEnv: the (opponent, won) keys injected by
    # the wrapper must survive Monitor and the process boundary.
    pool = _pool(tmp_path, [bridge_rush_script, defender_script])
    vec = make_opponent_vec_env(pool, n_envs=2, seed=1, start_method='fork')
    try:
        vec.reset()
        seen = False
        for _ in range(600):
            obs, rews, dones, infos = vec.step([[0, 0, 0], [0, 0, 0]])
            for i, info in enumerate(infos):
                if dones[i]:
                    assert info['opponent'] in (
                        'fixed:bridge_rush_script', 'fixed:defender_script')
                    assert info['won'] in (0, 1)
                    seen = True
            if seen:
                break
        assert seen, 'no episode finished within 600 steps'
    finally:
        vec.close()


# ---------------------------------------------------------------------------
# AdaptiveWeightCallback
# ---------------------------------------------------------------------------

def test_callback_pushes_priorities_to_children(tmp_path):
    pool = _pool(tmp_path, [bridge_rush_script, defender_script])
    vec = make_opponent_vec_env(pool, n_envs=2, seed=0, start_method='fork')
    try:
        cb = AdaptiveWeightCallback(update_every=1, alpha=0.3, floor=0.1)
        cb.model = _DummyModel(vec)
        cb.num_timesteps = 1
        cb.locals = {'infos': [{}], 'dones': [False]}
        cb._buffer = {'fixed:bridge_rush_script': (4, 8)}  # 0.5 winrate
        cb._on_step()  # 1 % 1 == 0 -> _update() runs and pushes

        prios = vec.env_method('get_priorities', indices=[0])[0]
        # target = max(0.1, 1-0.5) = 0.5; old = 1.0 -> 0.3*0.5 + 0.7*1.0
        assert prios['fixed:bridge_rush_script'] == pytest.approx(0.85)
        assert 'fixed:defender_script' not in prios  # no data -> untouched
    finally:
        vec.close()


def test_adaptive_callback_smoke_with_learn(tmp_path):
    pool = _pool(tmp_path, [bridge_rush_script, defender_script])
    vec = make_opponent_vec_env(pool, n_envs=2, seed=1, start_method='fork')
    cb = AdaptiveWeightCallback(update_every=16, alpha=0.3, floor=0.1)
    cb._buffer = {'fixed:bridge_rush_script': (4, 8)}  # 0.5 winrate
    try:
        model = PPO('MultiInputPolicy', vec, n_steps=8, batch_size=16, seed=0)
        model.learn(total_timesteps=32, callback=cb)
        # Real learn loop: callback received infos via update_locals() and
        # pushed the smoothed priority into the sub-env pools.
        prios = vec.env_method('get_priorities', indices=[0])[0]
        assert prios['fixed:bridge_rush_script'] == pytest.approx(0.85, abs=0.06)
    finally:
        vec.close()

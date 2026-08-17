"""Tests for the r2dreamer self-play opponent selection (old-PPO port).

Covers the pieces that make "self-play opponent evolution" work for the
R2-Dreamer / Clash Royale integration in ``r2dreamer/envs/cr_opponents.py``:

1. ``OpponentPool._recent_paths`` — the "recent self-play snapshots" bucket
   must pick the newest by *step count*, not by filename string order.
2. ``OpponentPool.pick`` — weighted draw returns a callable and records the
   opponent key (needed to attribute episode results).
3. ``adapt_priorities`` — winrate -> sampling priority smoothing rule.
4. ``build_opponent_pool`` — config-driven pool construction (recent_dir,
   hydra config fallback, fixed strategies, per-worker seed).
5. ``ClashRoyale`` with a pool — per-episode opponent picking, episode result
   reporting and priority push-back through the env hooks.
6. ``DreamerOpponent`` — a real r2dreamer checkpoint plays as the opponent
   (skipped when no trained checkpoint is available).
"""
import os
import sys

import numpy as np
import pytest

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_R2D = os.path.join(_BASE, 'r2dreamer')
if _R2D not in sys.path:
    sys.path.insert(0, _R2D)

from envs.cr import ClashRoyale  # noqa: E402
from envs.cr_opponents import (  # noqa: E402
    OpponentPool,
    adapt_priorities,
    build_opponent_pool,
    load_dreamer_opponent,
)


def _fixed_random_only_pool(tmp_path, **kw):
    return OpponentPool(
        base_checkpoints=[],
        recent_dir=str(tmp_path / 'snapshots'),
        n_recent=6,
        fixed_strategies=['random'],
        weights={'recent': 0.0, 'base': 0.0, 'fixed': 1.0},
        seed=0,
        **kw,
    )


def test_recent_paths_selects_newest_by_step_not_lexicographic(tmp_path):
    # Regression carried over from the old PPO pool: sorting filenames as
    # strings breaks once step counts cross a digit-length boundary
    # ('100000' < '20000' lexicographically).
    snap = tmp_path / 'snapshots'
    snap.mkdir()
    for step in [10000, 20000, 30000, 100000, 110000, 120000]:
        (snap / f'r2dreamer_{step}_steps.pt').touch()

    pool = OpponentPool(recent_dir=str(snap), prefix='r2dreamer', n_recent=3,
                        fixed_strategies=[], seed=0)
    paths = [os.path.basename(p) for p in pool._recent_paths()]
    assert paths == ['r2dreamer_30000_steps.pt', 'r2dreamer_100000_steps.pt',
                     'r2dreamer_110000_steps.pt', 'r2dreamer_120000_steps.pt'][-3:]


def test_pick_returns_callable_and_records_key(tmp_path):
    pool = _fixed_random_only_pool(tmp_path)
    for _ in range(5):
        opp = pool.pick()
        assert callable(opp)
        assert pool.last_key == 'fixed:random_strategy'

    # With a deterministic seed the RNG is reproducible.
    a = _fixed_random_only_pool(tmp_path)
    b = _fixed_random_only_pool(tmp_path)
    for _ in range(20):
        a.pick()
        b.pick()
        assert a.last_key == b.last_key


def test_adapt_priorities_rule():
    # Opponents that beat the agent get a high priority; crushed ones fall to
    # the floor but are never zeroed (old self_play.py::adapt_priorities).
    out = adapt_priorities({'a': (0, 10), 'b': (10, 10), 'c': (5, 10)},
                           old={}, alpha=1.0, floor=0.1)
    assert out['a'] == pytest.approx(1.0)
    assert out['b'] == pytest.approx(0.1)
    assert out['c'] == pytest.approx(0.5)
    # Smoothing: old priority is kept for candidates with no new games.
    out2 = adapt_priorities({'a': (0, 10)}, old={'a': 0.5}, alpha=0.5, floor=0.1)
    assert out2['a'] == pytest.approx(0.75)


def test_build_opponent_pool_from_config(tmp_path):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        'enabled': True,
        'prefix': 'r2dreamer',
        'n_recent': 4,
        'base_checkpoints': [str(tmp_path / 'base.pt')],
        'fixed_strategies': ['random', 'script:bridge_rush'],
        'weights': {'recent': 0.5, 'base': 0.3, 'fixed': 0.2},
        'deterministic': True,
        'device': 'cpu',
    })
    pool = build_opponent_pool(cfg, logdir=str(tmp_path / 'logdir'), rank=2, base_seed=0)
    assert pool.n_recent == 4
    assert pool.recent_dir == os.path.join(str(tmp_path / 'logdir'), 'snapshots')
    assert len(pool.fixed_strategies) == 2
    assert pool.base_checkpoints == [str(tmp_path / 'base.pt')]
    assert pool.weights == {'recent': 0.5, 'base': 0.3, 'fixed': 0.2}
    assert pool.deterministic is True

    # Per-worker seeds must differ so envs don't draw opponents in lock-step.
    p0 = build_opponent_pool(cfg, logdir=str(tmp_path / 'logdir'), rank=0, base_seed=7)
    p1 = build_opponent_pool(cfg, logdir=str(tmp_path / 'logdir'), rank=1, base_seed=7)
    assert p0._rng.random() != p1._rng.random()


def test_clashroyale_pool_episode_reporting(tmp_path):
    """ClashRoyale with a pool: per-episode pick + (opponent, won) reports."""
    pool = _fixed_random_only_pool(tmp_path)
    env = ClashRoyale(pool=pool, seed=0, speed=4.0)
    env.reset()

    # Two full episodes.  At speed 4 each env step = 2 s of battle time and the
    # simulator forces a draw at 180 s (~90 env steps), so 200 steps always
    # finishes a match.
    total = 0
    for _ in range(2):
        env.reset()
        done = False
        guard = 0
        while not done and guard < 200:
            action = np.zeros((5, 32, 18), dtype=np.float32)
            action[0, np.random.randint(5)] = 1.0
            action[1, np.random.randint(32)] = 1.0
            action[2, np.random.randint(18)] = 1.0
            _, _, done, _ = env.step(action)
            guard += 1
        assert done, 'episode should terminate (forced draw) within 200 steps'
        total += guard
        results = env.get_episode_results()
        assert len(results) == 1
        key, won = results[0]
        assert key == 'fixed:random_strategy'
        assert won in (0, 1)
    assert total > 0


def test_snapshot_pruning_keeps_newest_by_step(tmp_path):
    """Regression: save_snapshot must prune by step count, not filename string.

    Lexicographic order breaks once step counts cross a digit-length boundary
    ('r2dreamer_20000_steps.pt' > 'r2dreamer_100000_steps.pt'), which would
    delete the newest snapshots and leave stale ones feeding the opponent
    pool's "recent" bucket (same pitfall the old self_play.py fixed in
    _recent_paths).
    """
    from omegaconf import OmegaConf

    from buffer import Buffer
    from trainer import OnlineTrainer
    import tools

    tcfg = OmegaConf.create({
        'steps': 200000, 'pretrain': 0, 'eval_every': 1e4, 'eval_episode_num': 0,
        'video_pred_log': False, 'params_hist_log': False, 'batch_length': 64,
        'batch_size': 16, 'train_ratio': 64, 'action_repeat': 1,
        'update_log_every': 5e3, 'save_every': 1e4,
        'snapshot_every': 1e4, 'n_snapshots': 2,
    })
    bcfg = OmegaConf.create({'device': 'cpu', 'storage_device': 'cpu',
                             'batch_size': 16, 'batch_length': 64, 'max_size': 500})
    logdir = tmp_path / 'log'
    logdir.mkdir()
    trainer = OnlineTrainer(tcfg, Buffer(bcfg), tools.Logger(logdir), logdir, None, None)

    class FakeAgent:
        def state_dict(self):
            return {'w': 1}

    agent = FakeAgent()
    for s in [0, 20000, 100000, 110000]:
        trainer.save_snapshot(agent, s)
    kept = sorted(os.listdir(logdir / 'snapshots'))
    # Correct (step-keyed): the two newest are kept; 0 and 20000 pruned.
    # The lexicographic bug would instead delete 100000 and keep 20000.
    assert kept == ['r2dreamer_100000_steps.pt', 'r2dreamer_110000_steps.pt'], kept


def test_cr_opponents_import_does_not_chdir():
    """Regression: importing cr_opponents in the main process must not chdir.

    make_envs() imports envs.cr_opponents to build the SelfPlayController in
    the MAIN process.  envs.cr does ``os.chdir(src/clasher_new)`` at import
    time, so a transitive import of envs.cr would corrupt the trainer's
    relative logdir paths (metrics.jsonl / latest.pt).  This runs in a clean
    subprocess to assert the invariant from scratch.
    """
    import subprocess
    import sys

    code = (
        "import sys, os\n"
        f"sys.path.insert(0, {_R2D!r})\n"
        "cwd = os.getcwd()\n"
        "from envs.cr_opponents import SelfPlayController\n"
        "assert 'envs.cr' not in sys.modules, 'envs.cr was imported as a side effect'\n"
        "assert os.getcwd() == cwd, 'import changed the process CWD'\n"
        "print('NO_CHDIR_OK')\n"
    )
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert 'NO_CHDIR_OK' in out.stdout


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(_R2D, 'logdir', '0817_r2dreamer_cr', 'latest.pt')),
    reason='no trained r2dreamer checkpoint available',
)
def test_dreamer_opponent_plays(tmp_path):
    """A real r2dreamer checkpoint acts as the opponent (recurrent RSSM)."""
    ckpt = os.path.join(_R2D, 'logdir', '0817_r2dreamer_cr', 'latest.pt')
    opp = load_dreamer_opponent(ckpt, device='cpu', deterministic=True)
    assert callable(opp)

    from environment import CREnv

    env = CREnv(opponent_model=None)
    env.opponent = opp  # let CREnv drive the Dreamer opponent itself
    raw, _ = env.reset()
    opp.reset_episode()
    for _ in range(5):
        slot, y, x = opp(raw)
        assert 0 <= slot <= 4 and 0 <= y < 32 and 0 <= x < 18
        raw, _reward, _term, _trunc, _ = env.step((slot, y, x))
        if _term or _trunc:
            raw, _ = env.reset()
            opp.reset_episode()


def test_parallel_env_pool_and_controller(tmp_path):
    """End-to-end: ParallelEnv workers with a pool + SelfPlayController.

    This is the exact path training uses: make_env builds a per-worker pool in
    a spawned subprocess, episodes run through ``ParallelEnv.step``, and the
    main-process controller gathers (opponent, won) reports and pushes the
    adaptive priorities back into the worker's pool.
    """
    import torch
    from omegaconf import OmegaConf

    from envs import make_env
    from envs.cr_opponents import SelfPlayController
    from envs.parallel import ParallelEnv

    cfg = OmegaConf.create({
        'task': 'cr_clashroyale',
        'env_num': 1,
        'eval_episode_num': 0,
        'action_repeat': 1,
        'time_limit': 600,
        'train_ratio': 64,
        'seed': 0,
        'device': 'cpu',
        'logdir': str(tmp_path / 'logdir'),
        'opponent': 'random',
        'speed': 4.0,
        'opponent_pool': {
            'enabled': True, 'prefix': 'r2dreamer', 'n_recent': 6,
            'base_checkpoints': [], 'fixed_strategies': ['random'],
            'weights': {'recent': 0.6, 'base': 0.2, 'fixed': 0.2},
            'deterministic': False, 'device': 'cpu',
            'update_every': 5e4, 'alpha': 0.3, 'floor': 0.1, 'verbose': False,
        },
    })

    penv = ParallelEnv(lambda i: (lambda: make_env(cfg, i)), 1, 'cpu')
    _ = penv.observation_space  # triggers worker construction

    # Eval envs (train=False) keep the fixed opponent — the stable ruler —
    # even when the training pool is enabled.
    eval_env = make_env(cfg, 0, train=False)
    assert eval_env.unwrapped.pool is None
    assert eval_env.unwrapped._opponent == 'random'

    # (B, 55) flat multi one-hot action: slot=1, y=16, x=8.
    act = torch.zeros(1, 55)
    act[0, 1] = 1.0
    act[0, 5 + 16] = 1.0
    act[0, 5 + 32 + 8] = 1.0
    done = torch.ones(1, dtype=torch.bool)
    guard = 0
    while guard < 200:
        _trans, done = penv.step(act, done)
        guard += 1
        if bool(done[0]):
            break
    assert bool(done[0]), 'episode should terminate within 200 steps'

    # Main-process controller: gather the finished episode's (opponent, won),
    # compute priorities and push them back into the worker's pool.  With
    # update_every=1 the first update() must complete the full round-trip.
    ctrl = SelfPlayController(penv, update_every=1, verbose=True)
    ctrl.update(1)
    assert ctrl._priorities, 'controller should have computed priorities'
    key = list(ctrl._priorities)[0]
    assert key == 'fixed:random_strategy', key

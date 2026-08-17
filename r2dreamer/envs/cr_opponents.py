"""Self-play opponent selection for the Clash Royale R2-Dreamer integration.

Ports the "old PPO" self-play opponent mechanism from
``src/clasher_new/self_play.py`` onto the r2dreamer world-model stack:

- ``DreamerOpponent``: wraps a loaded R2-Dreamer agent (RSSM) into the
  stateless ``obs -> (slot, y, x)`` callable that ``CREnv.opponent_action``
  expects, threading the recurrent latent state (stoch / deter / prev_action)
  across the episode and converting raw simulator observations into the
  batched tensor dict ``Dreamer.act`` wants (same recipe as
  ``src/clasher_new/play_r2dreamer.py``).
- ``OpponentPool``: per-episode weighted mix of the *recent self-play
  snapshots of the training agent*, frozen *base* checkpoints and fixed
  scripted/random strategies, with optional per-candidate adaptive priorities
  derived from in-training winrates.  This is the old PPO
  ``OpponentPool`` (weighted draw + ``_recent_paths`` + adaptive priorities),
  generalised to load both r2dreamer ``.pt`` and stable-baselines3 ``.zip``
  checkpoints.
- ``SelfPlayController``: the main-process half of the adaptive scheme (old
  ``AdaptiveWeightCallback``).  It periodically pulls the per-episode
  ``(opponent, won)`` reports the env workers accumulate, converts them into
  sampling priorities via :func:`adapt_priorities`, and pushes the new
  priorities back into every worker's pool.

The "recent" bucket reads ``<logdir>/snapshots/r2dreamer_<step>_steps.pt``,
written by the trainer's periodic snapshot hook (``trainer.save_snapshot``).
Because those snapshots are checkpoints of the very policy being trained, the
opponent roster *evolves with the agent* — the self-play opponent evolution
the old PPO used.
"""

import contextlib
import glob
import io
import os
import random
import sys

import numpy as np

# The r2dreamer repo dir is only on sys.path when launched via train.py /
# play_r2dreamer.py (editable install exposes only envs*/optim*).  Make the
# lazy dreamer import robust regardless of the entry point.
_R2D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _R2D not in sys.path:
    sys.path.insert(0, _R2D)

# NOTE: do NOT import envs.cr at module level.  envs.cr chdirs into the
# simulator dir at import time; this module is imported by make_envs() in the
# MAIN process (to build the SelfPlayController), and that chdir would corrupt
# the trainer's relative logdir paths.  _GRID_SCALE is therefore imported
# lazily inside build_obs(), which only ever runs inside env workers.


# ---------------------------------------------------------------------------
# Fixed scripted opponents (mirror of src/clasher_new/self_play.py)
# ---------------------------------------------------------------------------

def _random_strategy(obs):
    # Avoid importing `random` from the game namespace twice; CREnv already
    # exports `random_strategy` which we resolve lazily in _resolve_fixed.
    from environment import random_strategy

    return random_strategy(obs)


def _resolve_fixed(spec):
    """Turn a config string (or a callable) into an ``obs -> (slot, y, x)`` fn.

    Accepts ``'random'``, ``'script:bridge_rush'`` / ``'script:bridge_rush_left'``
    / ``'script:defender'``, or an already-importable callable.
    """
    if callable(spec):
        return spec
    if not isinstance(spec, str):
        raise ValueError(f"cannot interpret fixed strategy: {spec!r}")
    if spec in ("random", "random_strategy"):
        from environment import random_strategy

        return random_strategy
    if spec.startswith("script:"):
        # Lazy import: self_play pulls in stable_baselines3.
        from self_play import bridge_rush_left_script, bridge_rush_script, defender_script

        scripts = {
            "script:bridge_rush": bridge_rush_script,
            "script:bridge_rush_left": bridge_rush_left_script,
            "script:defender": defender_script,
        }
        if spec in scripts:
            return scripts[spec]
        raise ValueError(f"unknown scripted opponent: {spec!r}")
    raise ValueError(f"unknown fixed opponent: {spec!r}")


# ---------------------------------------------------------------------------
# Observation / action conversion for a Dreamer opponent
# ---------------------------------------------------------------------------

def make_obs_spaces():
    """Replicate the gym spaces r2dreamer/envs/cr.py exposes to the agent."""
    import gymnasium as gym

    obs_space = gym.spaces.Dict(
        {
            "grid": gym.spaces.Box(low=0.0, high=1.0, shape=(32, 18, 15), dtype=np.float32),
            "hand": gym.spaces.Box(low=0.0, high=12.0, shape=(5,), dtype=np.float32),
            "elixir": gym.spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        }
    )
    act_space = gym.spaces.Box(low=0, high=1, shape=(5, 32, 18), dtype=np.float32)
    act_space.multi_discrete = True
    return obs_space, act_space


def build_obs(raw, device, is_first=False, is_last=False, is_terminal=False):
    """Turn CREnv.observe() raw dict into the batched tensor dict agent.act wants.

    Mirrors r2dreamer/envs/cr.py::_obs plus the trainer's (B, 1) lift for the
    bool scalars, with batch size 1.
    """
    import torch

    # Lazy import: see the module note — importing envs.cr chdirs the process,
    # which must never happen in the main process.
    from .cr import _GRID_SCALE  # noqa: E402

    grid = raw["grid"] / _GRID_SCALE
    grid = np.nan_to_num(grid, nan=0.0, posinf=1.0, neginf=0.0)
    obs = {
        "grid": torch.as_tensor(grid, dtype=torch.float32).unsqueeze(0),          # (1,32,18,15)
        "hand": torch.as_tensor(raw["hand"], dtype=torch.float32).unsqueeze(0),   # (1,5)
        "elixir": torch.as_tensor(raw["elixir"], dtype=torch.float32).unsqueeze(0),  # (1,1)
        "is_first": torch.tensor([[bool(is_first)]], dtype=torch.bool),
        "is_last": torch.tensor([[bool(is_last)]], dtype=torch.bool),
        "is_terminal": torch.tensor([[bool(is_terminal)]], dtype=torch.bool),
    }
    return {k: v.to(device) for k, v in obs.items()}


def decode_action(action):
    """55-dim multi one-hot -> (slot, y, x).  Same as envs/cr.py::step."""
    a = action[0].detach().cpu().numpy().reshape(-1)
    slot = int(np.argmax(a[0:5]))
    y = int(np.argmax(a[5:37]))
    x = int(np.argmax(a[37:55]))
    return slot, y, x


# ---------------------------------------------------------------------------
# Dreamer opponent
# ---------------------------------------------------------------------------

class DreamerOpponent:
    """A loaded R2-Dreamer agent as a ``CREnv`` opponent.

    ``CREnv.opponent_action`` calls the opponent with the raw player-1
    observation every decision step.  The Dreamer policy is recurrent (RSSM),
    so this wrapper owns the latent state and resets it at every episode
    boundary via :meth:`reset_episode` (called by ``ClashRoyale.reset`` when a
    new opponent is picked).
    """

    def __init__(self, agent, device="cpu", deterministic=False):
        self.agent = agent
        self.device = device
        self.deterministic = deterministic  # True = greedy (act eval=True)
        self.state = None
        self._first = True

    def reset_episode(self):
        self.state = self.agent.get_initial_state(1)
        self._first = True

    def __call__(self, raw_obs):
        if self.state is None:
            self.reset_episode()
        obs = build_obs(raw_obs, self.device, is_first=self._first)
        self._first = False
        action, self.state = self.agent.act(obs, self.state, eval=self.deterministic)
        return decode_action(action)


def _force_device(node, device):
    """Recursively override every ``device`` key in a loaded Hydra config."""
    from omegaconf import OmegaConf

    for k, v in node.items():
        if k == "device":
            node[k] = device
        elif OmegaConf.is_dict(v):
            _force_device(v, device)


def find_hydra_config(ckpt):
    """Locate the ``.hydra/config.yaml`` that describes a ``.pt`` checkpoint.

    Tries the checkpoint's own directory (standard logdir layout) and then one
    level up (snapshots live in ``<logdir>/snapshots/`` while the hydra config
    is at ``<logdir>/.hydra/config.yaml``).
    """
    for base in (os.path.dirname(ckpt), os.path.dirname(os.path.dirname(ckpt))):
        p = os.path.join(base, ".hydra", "config.yaml")
        if os.path.isfile(p):
            return p
    return None


def load_dreamer_opponent(ckpt, config_path=None, device="cpu", deterministic=False):
    """Build a :class:`DreamerOpponent` from a r2dreamer ``latest.pt`` snapshot.

    ``ckpt`` holds ``agent_state_dict`` (plus optional ``step``).  The agent is
    reconstructed from the Hydra config saved next to the checkpoint (same
    recipe as ``src/clasher_new/play_r2dreamer.py``), so architecture and the
    ``r2dreamer`` representation loss are taken from the training run itself.
    """
    import torch
    from dreamer import Dreamer
    from omegaconf import OmegaConf

    if config_path is None or not os.path.isfile(config_path):
        config_path = find_hydra_config(ckpt)
    if not config_path:
        raise FileNotFoundError(
            f"cannot find .hydra/config.yaml for r2dreamer checkpoint {ckpt}"
        )
    cfg = OmegaConf.load(config_path)
    _force_device(cfg.model, device)
    cfg.model.compile = False  # no torch.compile for opponent inference

    obs_space, act_space = make_obs_spaces()
    # The Dreamer constructor prints network shapes and parameter counts
    # (networks.py / dreamer.py).  Silence it here so loading one opponent per
    # worker doesn't spam the console — the main training agent's single
    # startup print in train.py is untouched.
    with contextlib.redirect_stdout(io.StringIO()):
        agent = Dreamer(cfg.model, obs_space, act_space).to(device)
        ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt_data["agent_state_dict"])
        agent.clone_and_freeze()  # refresh the frozen encoder/rssm/actor copies
        agent.eval()
    return DreamerOpponent(agent, device=device, deterministic=deterministic)


def _sb3_opponent(path, device="cpu", deterministic=False):
    """Load a stable-baselines3 PPO checkpoint as a stateless opponent."""
    from stable_baselines3 import PPO

    model = PPO.load(path, device=device)
    return lambda obs: model.predict(obs, deterministic=deterministic)[0]


def load_opponent(path, config_path=None, device="cpu", deterministic=False):
    """Load any checkpoint path: ``.zip`` -> SB3 PPO, ``.pt`` -> Dreamer."""
    if path.endswith(".zip"):
        return _sb3_opponent(path, device=device, deterministic=deterministic)
    return load_dreamer_opponent(path, config_path=config_path, device=device,
                                 deterministic=deterministic)


# ---------------------------------------------------------------------------
# OpponentPool (port of src/clasher_new/self_play.py::OpponentPool)
# ---------------------------------------------------------------------------

class OpponentPool:
    """Weighted mix of opponent candidates, one pick per training episode.

    Identical semantics to the old PPO pool:

    - ``recent``  : the ``n_recent`` most recent self-play snapshots of the
      training agent (``<recent_dir>/<prefix>_*_steps.pt``), sorted by the step
      count embedded in the filename — not lexicographically;
    - ``base``    : frozen checkpoints that already exist on disk;
    - ``fixed``   : non-learning strategies (random / scripts).

    Each ``pick()`` returns a callable for the next episode.  ``priorities``
    (candidate key -> sampling weight, fed by :class:`SelfPlayController`)
    scale each candidate's share — the old adaptive scheme.
    """

    def __init__(self, base_checkpoints=(), recent_dir=None, prefix="r2dreamer",
                 n_recent=6, fixed_strategies=None, weights=None, deterministic=False,
                 device="cpu", seed=None, priorities=None, run_config_path=None):
        self.base_checkpoints = list(base_checkpoints)
        self.recent_dir = recent_dir
        self.prefix = prefix
        self.n_recent = n_recent
        # Resolve string specs ('random', 'script:...') to callables up front so
        # pick() never hands out a bare string.
        self.fixed_strategies = [_resolve_fixed(s) for s in (fixed_strategies or [_random_strategy])]
        self.deterministic = deterministic
        self.device = device
        self.weights = dict(weights or {"recent": 0.6, "base": 0.2, "fixed": 0.2})
        self.priorities = dict(priorities or {})
        # Hydra config of the *current* run; used as fallback when a snapshot
        # has no config of its own (snapshots live one level under logdir).
        self.run_config_path = run_config_path
        self.last_key = None  # key of the opponent most recently handed out
        self._rng = random.Random(seed)
        self._cache = {}  # path -> callable (bounds memory)

    def _recent_paths(self):
        if not self.recent_dir or not os.path.isdir(self.recent_dir):
            return []
        pattern = os.path.join(self.recent_dir, f"{self.prefix}_*_steps.pt")
        suffix = "_steps.pt"

        def step(path):
            base = os.path.basename(path)
            return int(base[len(self.prefix) + 1:-len(suffix)])

        return sorted(glob.glob(pattern), key=step)[-self.n_recent:]

    def _load(self, path):
        if path not in self._cache:
            self._cache[path] = load_opponent(
                path,
                config_path=self.run_config_path,
                device=self.device,
                deterministic=self.deterministic,
            )
            while len(self._cache) > self.n_recent + len(self.base_checkpoints) + 2:
                del self._cache[next(iter(self._cache))]
        return self._cache[path]

    def child(self, rank, base_seed=None):
        """A fresh pool for one env subprocess (own RNG + model cache)."""
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
            run_config_path=self.run_config_path,
        )

    @staticmethod
    def _key(bucket, payload):
        if bucket == "fixed":
            name = payload.__name__ if callable(payload) else str(payload)
            return f"fixed:{name}"
        return f"{bucket}:{os.path.basename(payload)}"

    def set_priorities(self, priorities):
        """Replace the per-candidate sampling priorities (full dict)."""
        self.priorities = dict(priorities)

    def _priority(self, key):
        return self.priorities.get(key, 1.0)

    def _weighted_candidates(self):
        """Flat ``[(weight, kind, payload, key)]`` for one weighted draw.

        ``weight = base_bucket_weight / bucket_size * priority`` — within a
        bucket members are drawn in proportion to their priority, and a
        bucket's total share scales with the mean priority of its members.
        """
        entries = []
        recent = self._recent_paths()
        if recent:
            w = self.weights["recent"] / len(recent)
            for path in recent:
                key = self._key("recent", path)
                entries.append((w * self._priority(key), "path", path, key))
        if self.base_checkpoints:
            w = self.weights["base"] / len(self.base_checkpoints)
            for path in self.base_checkpoints:
                key = self._key("base", path)
                entries.append((w * self._priority(key), "path", path, key))
        if self.fixed_strategies:
            w = self.weights["fixed"] / len(self.fixed_strategies)
            for fn in self.fixed_strategies:
                key = self._key("fixed", fn)
                entries.append((w * self._priority(key), "fn", fn, key))
        return entries

    def pick(self):
        """Return one opponent callable for the next training episode."""
        entries = self._weighted_candidates()
        if not entries:
            self.last_key = "fixed:random_strategy"
            from environment import random_strategy

            return random_strategy
        total = sum(w for w, _, _, _ in entries)
        r = self._rng.random() * total
        for w, kind, payload, key in entries:
            r -= w
            if r <= 0:
                self.last_key = key
                return self._load(payload) if kind == "path" else payload
        _, kind, payload, key = entries[-1]
        self.last_key = key
        return self._load(payload) if kind == "path" else payload


def adapt_priorities(buffer, old, alpha=0.3, floor=0.1):
    """Turn ``{key: (wins, games)}`` into smoothed sampling priorities.

    ``priority = max(floor, 1 - winrate)`` — opponents that still beat the
    agent get a high priority (worth sampling more), ones it crushes fall to
    the floor (sampled rarely but never zeroed).  Exponentially smoothed
    against ``old``.  Pure function (same rule as the old self_play.py).
    """
    out = {}
    for key, (wins, games) in buffer.items():
        if games <= 0:
            continue
        target = max(floor, 1.0 - wins / games)
        out[key] = alpha * target + (1.0 - alpha) * old.get(key, 1.0)
    return out


# ---------------------------------------------------------------------------
# Main-process adaptive controller
# ---------------------------------------------------------------------------

class SelfPlayController:
    """Gather per-episode results from workers and adapt pool priorities.

    Every finished training episode the env worker records
    ``(opponent_key, won)`` (see ``ClashRoyale.step``).  This controller pulls
    those reports from all workers on every call, and every ``update_every``
    env-steps converts the per-candidate winrate into a sampling priority
    (:func:`adapt_priorities`) and pushes the new priorities back into every
    worker's pool — the old ``AdaptiveWeightCallback`` behaviour.
    """

    def __init__(self, train_envs, update_every=5e4, alpha=0.3, floor=0.1, verbose=False):
        self.train_envs = train_envs
        self.update_every = int(update_every)
        self.alpha = float(alpha)
        self.floor = float(floor)
        self.verbose = bool(verbose)
        self._buffer = {}       # candidate key -> [wins, games] since last update
        self._priorities = {}   # candidate key -> smoothed priority
        self._last = -1

    def update(self, train_step):
        if not hasattr(self.train_envs, "call_each"):
            return
        results = self.train_envs.call_each("get_episode_results")
        for per_env in results:
            for key, won in per_env or []:
                wins, games = self._buffer.get(key, (0, 0))
                self._buffer[key] = (wins + int(won), games + 1)
        if train_step > 0 and train_step != self._last and train_step % self.update_every == 0:
            self._last = train_step
            updated = adapt_priorities(self._buffer, self._priorities,
                                       alpha=self.alpha, floor=self.floor)
            self._priorities.update(updated)
            self._buffer = {}
            if updated:
                self.train_envs.call_each("set_pool_priorities", self._priorities)
            if self.verbose and updated:
                short = {k.split(":")[-1]: round(v, 2) for k, v in sorted(updated.items())}
                print("[self-play] priorities: " + ", ".join(f"{k}={v}" for k, v in short.items()))


# ---------------------------------------------------------------------------
# Factory (called inside each env worker)
# ---------------------------------------------------------------------------

def build_opponent_pool(pool_cfg, logdir="", rank=0, base_seed=0):
    """Build one :class:`OpponentPool` for a single env subprocess.

    ``pool_cfg`` is the ``env.opponent_pool`` Hydra section; ``logdir`` is the
    (already absolutized) run logdir — its ``snapshots/`` subdir is the
    self-play "recent" bucket and its ``.hydra/config.yaml`` the fallback
    config used to reconstruct Dreamer opponents.

    Opponent inference (RSSM forward) runs inside this worker.  Torch is
    pinned to a single thread so K workers don't fight over the CPU thread
    pool — the same choice the old ``make_opponent_vec_env`` made.
    """
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass

    recent_dir = getattr(pool_cfg, "recent_dir", "") or ""
    if not recent_dir and logdir:
        recent_dir = os.path.join(logdir, "snapshots")
    if recent_dir:
        recent_dir = os.path.abspath(recent_dir)

    run_config_path = None
    if logdir:
        p = os.path.join(os.path.abspath(logdir), ".hydra", "config.yaml")
        if os.path.isfile(p):
            run_config_path = p

    fixed_specs = list(getattr(pool_cfg, "fixed_strategies", None) or ["random"])
    base = list(getattr(pool_cfg, "base_checkpoints", None) or [])
    weights = dict(getattr(pool_cfg, "weights", None) or {"recent": 0.6, "base": 0.2, "fixed": 0.2})

    return OpponentPool(
        base_checkpoints=base,
        recent_dir=recent_dir,
        prefix=getattr(pool_cfg, "prefix", "r2dreamer"),
        n_recent=int(getattr(pool_cfg, "n_recent", 6)),
        fixed_strategies=[_resolve_fixed(s) for s in fixed_specs],
        weights=weights,
        deterministic=bool(getattr(pool_cfg, "deterministic", False)),
        device=getattr(pool_cfg, "device", "cpu"),
        seed=base_seed + rank,
        run_config_path=run_config_path,
    )

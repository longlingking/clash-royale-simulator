"""Clash Royale (clash-royale-simulator) environment adapter for Dreamer-style agents.

Bridges the gymnasium ``CREnv`` from the clash-royale-simulator project to the
interface r2dreamer (R2-Dreamer / DreamerV3) expects:

Observation dict (numpy arrays)::

    grid    (32, 18, 15) float32  -> CNN key (``cnn_keys: 'grid'``), channels normalized to ~[0, 1]
    hand    (5,) float32          -> MLP key, card ids (0..12) currently in hand
    elixir  (1,) float32          -> MLP key, 0..10
    is_first / is_last / is_terminal   bool scalars (required by trainer/agent)

Action: ``gym.spaces.Box(low=0, high=1, shape=(5, 32, 18))`` with
``.multi_discrete = True``. It is a flat multi one-hot vector (5 + 32 + 18 =
55 dims); the three segments ``(slot, y, x)`` are decoded by argmax per
segment and fed to ``CREnv.step`` as the ``(slot, y, x)`` tuple.

Notes
-----
- ``CREnv`` imports pygame (via ``new_visualization``) and its card modules
  read JSON files relative to CWD (``card_utils.py``). This module therefore
  pins ``SDL_VIDEODRIVER=dummy`` for headless pygame, adds ``src/clasher_new``
  to ``sys.path`` and chdirs into it before importing the game. Override the
  game directory with the ``CLASHER_SIM_DIR`` environment variable.
- The opponent can be the random baseline, one of the scripted strategies
  from ``self_play.py`` (``script:bridge_rush`` etc.), or a path to an
  existing stable-baselines3 PPO checkpoint.
"""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame (no display)
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")  # headless pygame (no audio -> no ALSA spam)

_HERE = os.path.dirname(os.path.abspath(__file__))
# This r2dreamer copy lives inside the clash-royale-simulator repo:
#   <repo>/r2dreamer/envs/cr.py  ->  <repo>/src/clasher_new
_REPO = os.path.dirname(os.path.dirname(_HERE))
# Game dependencies that cannot be pip-installed into the read-only conda env
# (e.g. fastcore) are vendored into <repo>/.pylibs via `pip install --target`.
_LIBS = os.path.join(_REPO, ".pylibs")
if _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)
_CLASHER = os.environ.get("CLASHER_SIM_DIR", os.path.join(_REPO, "src", "clasher_new"))
if _CLASHER not in sys.path:
    sys.path.insert(0, _CLASHER)
os.chdir(_CLASHER)

import numpy as np
import gymnasium as gym

from environment import CREnv, random_strategy  # noqa: E402

# Grid channel normalization. CREnv.observe() channel order is:
# [entity_id, player_id, elixir, card_type, speed, is_air, attacks_ground,
#  attacks_air, hp_left, hp_percentage, hit_speed, attack_range, sight_range,
#  damage, projectile_damage]. Divide each channel by its typical max
# magnitude so the CNN sees roughly [0, 1] inputs.
_GRID_SCALE = np.array(
    [12.0, 1.0, 10.0, 3.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float32,
)


class ClashRoyale(gym.Env):
    """gymnasium adapter of CREnv for r2dreamer (DreamerV3 / R2-Dreamer)."""

    def __init__(self, opponent="random", seed=0, speed=1.0):
        super().__init__()
        self._opponent = opponent
        self._env = CREnv(opponent_model=self._make_opponent(opponent), speed=speed)

        self.observation_space = gym.spaces.Dict(
            {
                "grid": gym.spaces.Box(low=0.0, high=1.0, shape=(32, 18, 15), dtype=np.float32),
                "hand": gym.spaces.Box(low=0.0, high=12.0, shape=(5,), dtype=np.float32),
                "elixir": gym.spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
                "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            }
        )
        # Flat multi one-hot action: [slot (5) | y (32) | x (18)].
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(5, 32, 18), dtype=np.float32)
        self.action_space.multi_discrete = True

    @staticmethod
    def _make_opponent(opponent):
        if opponent in (None, "", "random"):
            return random_strategy
        if isinstance(opponent, str) and opponent.startswith("script:"):
            # Lazy import: self_play pulls in stable_baselines3.
            from self_play import bridge_rush_left_script, bridge_rush_script, defender_script

            return {
                "script:bridge_rush": bridge_rush_script,
                "script:bridge_rush_left": bridge_rush_left_script,
                "script:defender": defender_script,
            }[opponent]
        if isinstance(opponent, str):  # path to a stable-baselines3 checkpoint
            from stable_baselines3 import PPO

            return PPO.load(opponent)
        return opponent

    def reset(self, *, seed=None, options=None):
        # NOTE: r2dreamer's wrapper chain (TimeLimit/Dtype/ParallelEnv) uses
        # the legacy gym API: reset() returns the bare observation dict (no
        # (obs, info) tuple) and step() returns a 4-tuple (obs, reward, done,
        # info).
        obs, _ = self._env.reset(seed=seed, options=options)
        return self._obs(obs, is_first=True)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        slot = int(np.argmax(action[0:5]))
        y = int(np.argmax(action[5:37]))
        x = int(np.argmax(action[37:55]))
        obs, reward, terminated, truncated, _ = self._env.step((slot, y, x))
        is_last = bool(terminated or truncated)
        return (
            self._obs(obs, is_first=False, is_last=is_last, is_terminal=bool(terminated)),
            float(reward),
            is_last,
            {},
        )

    def _obs(self, raw, is_first=False, is_last=False, is_terminal=False):
        grid = raw["grid"] / _GRID_SCALE
        # Defensive: hp_left = log(hp)/10 can be -inf when an entity's hp hits
        # 0, and spell entities can divide by zero -> keep the world model fed
        # with finite values only.
        grid = np.nan_to_num(grid, nan=0.0, posinf=1.0, neginf=0.0)
        return {
            "grid": grid.astype(np.float32),
            "hand": raw["hand"].astype(np.float32),
            "elixir": raw["elixir"].astype(np.float32),
            "is_first": np.bool_(is_first),
            "is_last": np.bool_(is_last),
            "is_terminal": np.bool_(is_terminal),
        }

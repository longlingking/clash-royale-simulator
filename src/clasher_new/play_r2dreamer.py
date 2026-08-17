"""Watch the R2-Dreamer agent (world model) play in a live pygame window.

``play.py`` can only watch stable-baselines3 PPO checkpoints (it calls
``PPO.load``).  The R2-Dreamer checkpoints produced by ``train_cr.py`` are
Dreamer/RSSM world-model state dicts (``logdir/<date>_r2dreamer_cr/latest.pt``),
so this script mirrors ``play.py`` but:

* reconstructs the Dreamer agent from the training Hydra config
  (``<logdir>/.hydra/config.yaml``) and loads ``agent_state_dict``;
* threads the recurrent latent state (stoch / deter / prev_action) through
  ``agent.act(obs, state, eval=True)`` on every decision — the PPO loop in
  ``play.py`` is stateless and cannot do this;
* decodes the actor's flat 55-dim multi one-hot (5 slot | 32 y | 18 x) back to
  the ``(slot, y, x)`` tuple ``CREnv.step`` expects (same decoding as
  ``r2dreamer/envs/cr.py``);
* builds the observation dict exactly like ``envs/cr.py`` (grid normalized by
  ``_GRID_SCALE``, nan->0, hand/elixir float32, is_first/is_last/is_terminal).

Window controls (handled by ``Visualizer.process_events``):
    ESC / close  quit
    SPACE        pause / unpause (between decisions)
    1..5         live speed (same as --speed)

Usage:
    python play_r2dreamer.py                                          # latest 0817 logdir vs random
    python play_r2dreamer.py --opponent bridge_rush --speed 2 --games 3
    python play_r2dreamer.py --checkpoint /path/to/latest.pt --stochastic
"""
import argparse
import os
import sys
import time

import numpy as np

# Card modules open gamedata.json relative to CWD, so pin the working dir to
# this script's location before importing anything game-related.
_BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_BASE)

# r2dreamer is vendored next to the repo; dreamer.py imports networks/rssm/tools.
_R2D = os.path.abspath(os.path.join(_BASE, "..", "..", "r2dreamer"))
if _R2D not in sys.path:
    sys.path.insert(0, _R2D)

import pygame  # noqa: E402  (after os.chdir, before the r2dreamer imports below)
import torch  # noqa: E402

from dreamer import Dreamer  # noqa: E402
from environment import CREnv, random_strategy  # noqa: E402
from self_play import (  # noqa: E402
    bridge_rush_left_script,
    bridge_rush_script,
    defender_script,
)
from omegaconf import OmegaConf  # noqa: E402

OPPONENTS = {
    'random': random_strategy,
    'bridge_rush': bridge_rush_script,
    'bridge_rush_left': bridge_rush_left_script,
    'defender': defender_script,
}

# Same grid channel normalization as r2dreamer/envs/cr.py.
_GRID_SCALE = np.array(
    [12.0, 1.0, 10.0, 3.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float32,
)

_DEFAULT_CKPT = os.path.join(
    _BASE, "..", "..", "r2dreamer", "logdir", "0817_r2dreamer_cr", "latest.pt"
)


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

    Mirrors r2dreamer/envs/cr.py::_obs + the trainer's (B, 1) lift for the
    bool scalars, with batch size 1.
    """
    grid = raw["grid"] / _GRID_SCALE
    grid = np.nan_to_num(grid, nan=0.0, posinf=1.0, neginf=0.0)
    obs = {
        "grid": torch.as_tensor(grid, dtype=torch.float32).unsqueeze(0),        # (1,32,18,15)
        "hand": torch.as_tensor(raw["hand"], dtype=torch.float32).unsqueeze(0),  # (1,5)
        "elixir": torch.as_tensor(raw["elixir"], dtype=torch.float32).unsqueeze(0),  # (1,1)
        "is_first": torch.tensor([[bool(is_first)]], dtype=torch.bool),
        "is_last": torch.tensor([[bool(is_last)]], dtype=torch.bool),
        "is_terminal": torch.tensor([[bool(is_terminal)]], dtype=torch.bool),
    }
    return {k: v.to(device) for k, v in obs.items()}


def decode_action(action):
    """55-dim multi one-hot -> (slot, y, x). Same as envs/cr.py::step."""
    a = action[0].detach().cpu().numpy().reshape(-1)
    slot = int(np.argmax(a[0:5]))
    y = int(np.argmax(a[5:37]))
    x = int(np.argmax(a[37:55]))
    return slot, y, x


def main():
    ap = argparse.ArgumentParser(
        description='Watch the R2-Dreamer agent play a fixed scripted opponent.')
    ap.add_argument('--checkpoint', default=os.path.normpath(_DEFAULT_CKPT),
                    help='r2dreamer latest.pt to watch '
                         '(default r2dreamer/logdir/0817_r2dreamer_cr/latest.pt)')
    ap.add_argument('--config', default=None,
                    help='Hydra config.yaml; defaults to <logdir>/.hydra/config.yaml '
                         'next to the checkpoint')
    ap.add_argument('--opponent', default='random', choices=list(OPPONENTS),
                    help='fixed scripted opponent (default random)')
    ap.add_argument('--speed', type=int, default=1,
                    help='sim speed, 1 = realtime (default 1)')
    ap.add_argument('--games', type=int, default=1,
                    help='games to play before exiting (default 1)')
    ap.add_argument('--stochastic', action='store_true',
                    help='sample actions instead of greedy (for variety)')
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                    help='inference device (default cpu)')
    args = ap.parse_args()

    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt):
        sys.exit(f'checkpoint not found: {ckpt}')

    logdir = os.path.dirname(ckpt)
    config_path = args.config or os.path.join(logdir, '.hydra', 'config.yaml')
    if not os.path.isfile(config_path):
        sys.exit(f'hydra config not found: {config_path} (pass --config)')
    cfg = OmegaConf.load(config_path)

    def _force_device(node, device):
        # The saved config pins device: cuda on several sub-modules
        # (rssm/encoder/actor/...), which breaks CPU playback.
        for k, v in node.items():
            if k == 'device':
                node[k] = device
            elif OmegaConf.is_dict(v):
                _force_device(v, device)

    _force_device(cfg.model, args.device)
    cfg.model.compile = False  # no torch.compile for playback
    if not args.stochastic:
        cfg.model.act_entropy = 0.0  # not used by act(eval=True) anyway; keep as-is

    torch.set_num_threads(1)
    device = torch.device(args.device)

    obs_space, act_space = make_obs_spaces()
    print(f'building agent (rep_loss={cfg.model.rep_loss}) ...')
    agent = Dreamer(cfg.model, obs_space, act_space).to(device)
    ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
    agent.load_state_dict(ckpt_data["agent_state_dict"])
    # refresh frozen copies (encoder/rssm/actor/...) from the loaded weights
    agent.clone_and_freeze()
    agent.eval()
    print(f'loaded checkpoint -> {ckpt}  (step {ckpt_data.get("step")})')

    opponent = OPPONENTS[args.opponent]
    print(f'playing vs       -> {args.opponent}  (speed {args.speed}x, '
          f'{"stochastic" if args.stochastic else "greedy"})')

    env = CREnv(opponent_model=None, visualize=True, speed=args.speed)
    env.opponent = opponent

    try:
        for game in range(args.games):
            try:
                raw, _ = env.reset()
            except pygame.error as e:
                print(f'pygame could not open a window: {e}')
                print('play_r2dreamer.py needs a display (local screen, or ssh -X).')
                return
            env.visualizer.speed = args.speed  # start at the requested speed

            # Recurrent state: zeroed at episode start (is_first=True resets it).
            state = agent.get_initial_state(1)

            done = False
            is_first = True
            while not done:
                env.visualizer.process_events()
                if not env.visualizer.running:
                    print('window closed, stopping')
                    return
                if env.visualizer.paused:
                    time.sleep(0.05)
                    continue
                env.speed = env.visualizer.speed  # live 1-5 speed keys

                obs = build_obs(raw, device, is_first=is_first)
                action, state = agent.act(obs, state, eval=not args.stochastic)
                slot, y, x = decode_action(action)
                raw, _reward, terminated, truncated, _info = env.step((slot, y, x))
                is_first = False
                done = terminated or truncated

            winner = env.battle.winner
            if winner == 0:
                result = 'WIN'
            elif winner == 1:
                result = 'LOSS'
            else:
                result = 'draw'
            print(f'game {game + 1}/{args.games}: {result}')
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()

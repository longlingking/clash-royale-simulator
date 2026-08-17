import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# card_utils.py opens gamedata.json relative to CWD, so pin the working
# directory to this script's location before importing anything game-related.
_BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_BASE)

from environment import random_strategy, entity_names

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback

from self_play import (
    AdaptiveWeightCallback,
    BestWeightCallback,
    OpponentPool,
    make_opponent_vec_env,
    bridge_rush_left_script,
    bridge_rush_script,
    defender_script,
)


class CRFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.embedding_dim = 8
        self.entity_embedding = nn.Embedding(len(entity_names), self.embedding_dim)
        self.in_channels = 13 + self.embedding_dim + 4
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, 32, 18)
            cnn_out = self.cnn(dummy).shape[1]
        self.fc = nn.Linear(cnn_out + 5 * self.embedding_dim + 1, features_dim)

    def forward(self, observation):
        """
        Gets the observation, use the embedding (dim=8) to expand the channels, then use one-hot to further expand the channels.
        The code is ugly but should do the work.
        """
        grid = observation['grid']  # (B, 32, 18, 15)
        hand = observation['hand'].long()  # (B, 5)
        elixir = observation['elixir']

        card_ids = grid[..., 0].long()
        card_vecs = self.entity_embedding(card_ids)

        rest = grid[..., 1:]  # (B, 32, 18, 14)
        x = torch.cat([rest, card_vecs], dim=-1)  # (B, 32, 18, 14+EMBED)
        card_type = x[..., 0].long()  # (B, 32, 18)
        card_type_oh = F.one_hot(card_type, num_classes=4).float()  # (B, 32, 18, 4)
        rest = x[..., 1:]
        x = torch.cat([rest, card_type_oh], dim=-1)
        x = x.permute(0, 3, 1, 2).float()  # (B, C, 32, 18)

        grid_feat = self.cnn(x)

        hand_feat = self.entity_embedding(hand).flatten(1)  # (B, 5*EMBED)
        combined = torch.cat([grid_feat, hand_feat, elixir.float()], dim=1)
        return torch.relu(self.fc(combined))


def make_eval_crowd(base_dir):
    """The FIXED opponents used for best-model selection. Never changes during
    a run, so the winrate numbers stay comparable across evaluations.

    Two of them are the already-trained checkpoints on disk (frozen here),
    the rest are non-learning baselines.
    """
    return {
        'random': random_strategy,
        'bridge_rush': bridge_rush_script,
        'bridge_rush_left': bridge_rush_left_script,
        'defender': defender_script,
        'cr_discrete': os.path.join(base_dir, 'cr_discrete.zip'),
        'cr_checkpoint': os.path.join(base_dir, 'cr_checkpoint.zip'),
    }


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=1_000_000,
                    help='total training timesteps (small for smoke runs)')
    ap.add_argument('--save-path', type=str, default='cr_discrete.zip',
                    help='where the trained model is saved on exit. '
                         'NOTE: the default overwrites the cr_discrete baseline; '
                         'use a different name (e.g. cr_trained.zip) to keep it.')
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='policy device. The sim + opponent inference always run '
                         'on CPU in the subprocess envs; this only moves the '
                         'agent policy + PPO updates (which dominate wall-clock). '
                         'GPU makes updates ~16x faster than 1 CPU thread.')
    ap.add_argument('--adap-update-every', type=int, default=50_000,
                    help='env-steps between adaptive opponent-priority updates '
                         '(future.md #2)')
    ap.add_argument('--adap-alpha', type=float, default=0.3,
                    help='exponential smoothing for adaptive priorities')
    ap.add_argument('--adap-floor', type=float, default=0.1,
                    help='minimum adaptive priority (forgetting guard)')
    args = ap.parse_args()

    log_dir = os.path.join(_BASE, 'cr_logs')
    os.makedirs(log_dir, exist_ok=True)

    # Single-threaded inference: with K subprocess envs running the sim, the
    # main process should not fight them for cores on a 0.62M-param forward
    # pass (see benchmark_env.py).
    torch.set_num_threads(1)

    # Self-play opponent pool: mostly recent snapshots of ourselves, some old
    # checkpoints, some fixed baselines. Opponents never receive gradients.
    pool = OpponentPool(
        base_checkpoints=[
            os.path.join(_BASE, 'cr_discrete.zip'),
            os.path.join(_BASE, 'cr_checkpoint.zip'),
        ],
        recent_dir=log_dir,
        prefix='cr',
        n_recent=6,
        fixed_strategies=[
            random_strategy,
            bridge_rush_script,
            bridge_rush_left_script,
            defender_script,
        ],
        weights={'recent': 0.6, 'base': 0.2, 'fixed': 0.2},
        device='cpu',
    )

    # Per-env opponent sampling (future.md #1): each sub-env re-picks its
    # opponent from the pool at every episode reset, so a single PPO batch
    # mixes trajectories from all opponent types in proportion to the pool
    # weights — replacing the old window-level OpponentSwapCallback.
    n_envs = 8
    vec_env = make_opponent_vec_env(pool, n_envs=n_envs)

    # Keep the PPO batch the same size as the old single-env setup
    # (2048 = n_envs * n_steps) so the only behavioural change is the mix.
    n_steps = 2048 // n_envs

    # CheckpointCallback.save_freq counts *vec-steps* (one vec-step = n_envs
    # env-steps), so divide by n_envs to keep the intended cadence of one
    # checkpoint per 10_000 env-steps — otherwise the "recent self-play
    # opponent" pool ends up 8x coarser than the single-env design.
    checkpoint_every_env_steps = 10_000
    save_freq = max(checkpoint_every_env_steps // n_envs, 1)

    # The policy + PPO updates go on args.device (GPU when available); the
    # subprocess envs keep running the sim on CPU.
    model = PPO.load(os.path.join(_BASE, 'cr_discrete'), env=vec_env,
                     n_steps=n_steps, device=args.device,
                     tensorboard_log=os.path.join(_BASE, 'tb_logs'))

    callbacks = [
        CheckpointCallback(save_freq=save_freq, save_path=log_dir, name_prefix='cr'),
        BestWeightCallback(
            opponents=make_eval_crowd(_BASE),
            eval_every=100_000,
            n_games=20,
            best_path=os.path.join(_BASE, 'best_model.zip'),
            eval_at_start=True,
            verbose=1,
        ),
        AdaptiveWeightCallback(
            update_every=args.adap_update_every,
            alpha=args.adap_alpha,
            floor=args.adap_floor,
            verbose=1,
        ),
    ]
    try:
        model.learn(total_timesteps=args.steps, reset_num_timesteps=False,
                    tb_log_name='cr', callback=callbacks)
    finally:
        print('Saving model.')
        model.save(os.path.join(_BASE, args.save_path))
        vec_env.close()  # reap the 8 subprocess envs (avoids a hang on exit)

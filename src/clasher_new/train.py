import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# card_utils.py opens gamedata.json relative to CWD, so pin the working
# directory to this script's location before importing anything game-related.
_BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_BASE)

from environment import CREnv, random_strategy, entity_names

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback

from self_play import (
    BestWeightCallback,
    OpponentPool,
    OpponentSwapCallback,
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
    log_dir = os.path.join(_BASE, 'cr_logs')
    os.makedirs(log_dir, exist_ok=True)

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

    env = CREnv(opponent_model=pool.pick())

    model = PPO.load(os.path.join(_BASE, 'cr_discrete'), env=env,
                     tensorboard_log=os.path.join(_BASE, 'tb_logs'))

    callbacks = [
        CheckpointCallback(save_freq=10_000, save_path=log_dir, name_prefix='cr'),
        OpponentSwapCallback(env=env, pool=pool, swap_every=2048),
        BestWeightCallback(
            opponents=make_eval_crowd(_BASE),
            eval_every=100_000,
            n_games=20,
            best_path=os.path.join(_BASE, 'best_model.zip'),
            eval_at_start=True,
            verbose=1,
        ),
    ]
    try:
        model.learn(total_timesteps=1_000_000, reset_num_timesteps=False,
                    tb_log_name='cr', callback=callbacks)
    finally:
        print('Saving model.')
        model.save(os.path.join(_BASE, 'cr_discrete'))

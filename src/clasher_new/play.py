"""Watch a trained agent play a fixed scripted opponent in a live pygame window.

The agent is ``--model`` (default ``best_model.zip``, the BestWeightCallback
pick) and the opponent is a fixed, non-learning script (default ``random``).
The agent plays player 0 with greedy/deterministic actions, exactly like
``compare_weights.py`` / ``evaluate_agent``, so what you see is what the eval
numbers describe.

Window controls (handled by ``Visualizer.process_events``, which the plain
training loop never calls but this script does):
    ESC / close  quit
    SPACE        pause / unpause (between decisions)
    1..5         live speed (same as --speed)

Usage:
    python play.py                          # best_model.zip vs random
    python play.py --model cr_adap.zip --opponent bridge_rush
    python play.py --opponent defender --speed 2 --games 3
"""
import argparse
import os
import time

import torch

# card_utils.py opens gamedata.json relative to CWD, so pin the working
# directory to this script's location before importing anything game-related.
_BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_BASE)

import pygame  # noqa: E402  (needs to come after os.chdir only for card_utils)
from stable_baselines3 import PPO  # noqa: E402

from environment import CREnv, random_strategy  # noqa: E402
from self_play import (  # noqa: E402
    bridge_rush_left_script,
    bridge_rush_script,
    defender_script,
)

OPPONENTS = {
    'random': random_strategy,
    'bridge_rush': bridge_rush_script,
    'bridge_rush_left': bridge_rush_left_script,
    'defender': defender_script,
}


def main():
    ap = argparse.ArgumentParser(
        description='Watch a trained agent play a fixed scripted opponent.')
    ap.add_argument('--model', default='best_model.zip',
                    help='checkpoint to watch (default best_model.zip)')
    ap.add_argument('--opponent', default='random', choices=list(OPPONENTS),
                    help='fixed scripted opponent (default random)')
    ap.add_argument('--speed', type=int, default=1,
                    help='sim speed, 1 = realtime (default 1)')
    ap.add_argument('--games', type=int, default=1,
                    help='games to play before exiting (default 1)')
    ap.add_argument('--stochastic', action='store_true',
                    help='sample actions instead of greedy (for variety)')
    args = ap.parse_args()

    # Small model: single-thread inference is faster (and quieter) than having
    # every core fight over the forward pass.
    torch.set_num_threads(1)

    print(f'loading model    -> {args.model}')
    model = PPO.load(args.model, device='cpu')
    opponent = OPPONENTS[args.opponent]
    print(f'playing vs       -> {args.opponent}  (speed {args.speed}x)')

    env = CREnv(opponent_model=None, visualize=True, speed=args.speed)
    env.opponent = opponent

    try:
        for game in range(args.games):
            try:
                obs, _ = env.reset()
            except pygame.error as e:
                print(f'pygame could not open a window: {e}')
                print('play.py needs a display (local screen, or ssh -X).')
                return
            env.visualizer.speed = args.speed  # start at the requested speed

            done = False
            while not done:
                # The CREnv render loop never calls process_events(), so the
                # window would be unclosable without this.
                env.visualizer.process_events()
                if not env.visualizer.running:
                    print('window closed, stopping')
                    return
                if env.visualizer.paused:
                    time.sleep(0.05)
                    continue
                env.speed = env.visualizer.speed  # live 1-5 speed keys

                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, _reward, terminated, truncated, _info = env.step(action)
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

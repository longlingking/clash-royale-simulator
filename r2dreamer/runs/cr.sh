#!/bin/bash
# Train R2-Dreamer (world model) on the Clash Royale simulator.
#
# Usage:
#   bash runs/cr.sh                         # defaults below
#   bash runs/cr.sh 100000                  # custom steps
#
# CPU-only by default (no GPU on this machine). Opponent options in
# configs/env/cr.yaml: 'random' | 'script:bridge_rush' |
# 'script:bridge_rush_left' | 'script:defender' | <sb3 checkpoint path>.

DATE=$(date +%m%d)
STEPS=${1:-500000}
METHOD=r2dreamer

python train.py \
    env=cr \
    env.steps=$STEPS \
    logdir=logdir/${DATE}_${METHOD}_cr \
    model.rep_loss=${METHOD} \
    model.compile=False \
    device=cpu \
    batch_size=16 \
    batch_length=64 \
    trainer.train_ratio=64 \
    seed=0

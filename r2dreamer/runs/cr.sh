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

# r2dreamer uses bf16 autocast for the CPU update step; oneDNN's bf16
# backward path crashes on avx2_vnni_2 CPUs (e.g. Intel Core Ultra 200
# series): "DNNL does not support bf16/f16 backward". Cap ISA to AVX2.
export ONEDNN_MAX_CPU_ISA=AVX2

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

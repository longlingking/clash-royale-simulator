#!/bin/bash
# Train R2-Dreamer (world model) on the Clash Royale simulator.
#
# Usage:
#   bash runs/cr.sh                         # CPU, 1 env (default)
#   bash runs/cr.sh 100000                  # CPU, custom steps
#   GPU=1 bash runs/cr.sh 2000              # GPU (cuda) + 8 parallel CPU sim workers
#   GPU=1 ENV_NUM=16 bash runs/cr.sh 2000   # GPU, custom parallelism
#   GPU=1 COMPILE=1 bash runs/cr.sh 2000    # GPU + torch.compile (needs triton)
#
# Architecture: the simulator always runs in CPU subprocesses (ParallelEnv
# workers, one per env); only the agent policy inference + world-model
# updates run on the device. GPU=1 moves those to cuda (pin_memory + async
# H2D are already handled by envs/parallel.py / trainer.py).
#
# Opponent options in configs/env/cr.yaml: 'random' | 'script:bridge_rush' |
# 'script:bridge_rush_left' | 'script:defender' | <sb3 checkpoint path>.

DATE=$(date +%m%d)
STEPS=${1:-500000}
METHOD=r2dreamer
GPU=${GPU:-0}
ENV_NUM=${ENV_NUM:-8}      # parallel sim workers (CPU subprocesses)
COMPILE=${COMPILE:-0}      # torch.compile the update fn (GPU only; needs triton)

# r2dreamer uses bf16 autocast for the CPU update step; oneDNN's bf16
# backward path crashes on avx2_vnni_2 CPUs (e.g. Intel Core Ultra 200
# series): "DNNL does not support bf16/f16 backward". Cap ISA to AVX2.
# Harmless on GPU: autocast there is fp16 via cuDNN, oneDNN only touches CPU ops.
export ONEDNN_MAX_CPU_ISA=AVX2

if [ "$GPU" = "1" ]; then
    DEV_ARGS="device=cuda env.env_num=$ENV_NUM model.compile=$COMPILE"
else
    DEV_ARGS="device=cpu model.compile=False env.env_num=1"
fi

python train.py \
    env=cr \
    env.steps=$STEPS \
    logdir=logdir/${DATE}_${METHOD}_cr \
    model.rep_loss=${METHOD} \
    $DEV_ARGS \
    batch_size=16 \
    batch_length=64 \
    trainer.train_ratio=64 \
    seed=0

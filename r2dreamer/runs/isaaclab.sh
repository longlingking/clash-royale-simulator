#!/bin/bash

# ==== Settings ====
GPU_ID=0
DATE=$(date +%m%d)
SEED_START=0
SEED_END=400
SEED_STEP=100
MODAL=vision  # proprio/vision
METHOD=r2dreamer

# ==== Tasks ====
# Each entry is "isaaclab_<GymID>" — the gym registration ID is extracted
# automatically by the code (everything after "isaaclab_").
# Any IsaacLab task registered in gymnasium can be used here.
tasks=(
    isaaclab_Isaac-Cartpole-RGB-Camera-Direct-v0
)

# ==== Loop ====
for task in "${tasks[@]}"
do
    # Strip "isaaclab_" prefix for logdir readability.
    short_name=${task#isaaclab_}
    for seed in $(seq $SEED_START $SEED_STEP $SEED_END)
    do
        CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
            env=isaaclab_${MODAL} \
            env.task=$task \
            logdir=logdir/${DATE}_${METHOD}_isaaclab_${short_name}_$seed \
            model.compile=True \
            device=cuda:0 \
            buffer.storage_device=cuda:0 \
            model.rep_loss=${METHOD} \
            seed=$seed
    done
done

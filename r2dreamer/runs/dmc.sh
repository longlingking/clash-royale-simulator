#!/bin/bash

# ==== Settings ====
GPU_ID=0
DATE=$(date +%m%d) # auto complete
SEED_START=0
SEED_END=400
SEED_STEP=100
MODAL=vision # vision/proprio
METHOD=r2dreamer

# ==== Tasks ====
tasks=(
    dmc_acrobot_swingup
    dmc_ball_in_cup_catch
    dmc_cartpole_balance
    dmc_cartpole_balance_sparse
    dmc_cartpole_swingup
    dmc_cartpole_swingup_sparse
    dmc_cheetah_run
    dmc_finger_spin
    dmc_finger_turn_easy
    dmc_finger_turn_hard
    dmc_hopper_hop
    dmc_hopper_stand
    dmc_pendulum_swingup
    dmc_quadruped_run
    dmc_quadruped_walk
    dmc_reacher_easy
    dmc_reacher_hard
    dmc_walker_run
    dmc_walker_stand
    dmc_walker_walk
)

# ==== Loop ====
for task in "${tasks[@]}"
do
    for seed in $(seq $SEED_START $SEED_STEP $SEED_END)
    do
        CUDA_VISIBLE_DEVICES=$GPU_ID MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU_ID python train.py \
            env=dmc_${MODAL} \
            env.task=$task \
            logdir=logdir/${DATE}_${METHOD}_${task#dmc_}_$seed \
            model.compile=True \
            device=cuda:0 \
            buffer.storage_device=cuda:0 \
            model.rep_loss=${METHOD} \
            seed=$seed
    done
done

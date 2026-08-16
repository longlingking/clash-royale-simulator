#!/bin/bash

# ==== Settings ====
GPU_ID=0
DATE=$(date +%m%d) # auto complete
SEED_START=0
SEED_END=400
SEED_STEP=100
METHOD=r2dreamer

# ==== Tasks ====
tasks=(
        "metaworld_assembly"
        "metaworld_basketball"
        "metaworld_bin-picking"
        "metaworld_box-close"
        "metaworld_button-press-topdown"
        "metaworld_button-press-topdown-wall"
        "metaworld_button-press"
        "metaworld_button-press-wall"
        "metaworld_coffee-button"
        "metaworld_coffee-pull"
        "metaworld_coffee-push"
        "metaworld_dial-turn"
        "metaworld_disassemble"
        "metaworld_door-close"
        "metaworld_door-lock"
        "metaworld_door-open"
        "metaworld_door-unlock"
        "metaworld_hand-insert"
        "metaworld_drawer-close"
        "metaworld_drawer-open"
        "metaworld_faucet-open"
        "metaworld_faucet-close"
        "metaworld_hammer"
        "metaworld_handle-press-side"
        "metaworld_handle-press"
        "metaworld_handle-pull-side"
        "metaworld_handle-pull"
        "metaworld_lever-pull"
        "metaworld_pick-place-wall"
        "metaworld_pick-out-of-hole"
        "metaworld_pick-place"
        "metaworld_plate-slide"
        "metaworld_plate-slide-side"
        "metaworld_plate-slide-back"
        "metaworld_plate-slide-back-side"
        "metaworld_peg-insert-side"
        "metaworld_peg-unplug-side"
        "metaworld_soccer"
        "metaworld_stick-push"
        "metaworld_stick-pull"
        "metaworld_push"
        "metaworld_push-wall"
        "metaworld_push-back"
        "metaworld_reach"
        "metaworld_reach-wall"
        "metaworld_shelf-place"
        "metaworld_sweep-into"
        "metaworld_sweep"
        "metaworld_window-open"
        "metaworld_window-close"
)

# ==== Loop ====
for task in "${tasks[@]}"
do
    for seed in $(seq $SEED_START $SEED_STEP $SEED_END)
    do
        CUDA_VISIBLE_DEVICES=$GPU_ID MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU_ID python train.py \
            env=metaworld \
            env.task=$task \
            logdir=logdir/${DATE}_${METHOD}_${task#metaworld_}_$seed \
            model.compile=True \
            device=cuda:0 \
            buffer.storage_device=cuda:0 \
            model.rep_loss=${METHOD} \
            seed=$seed
    done
done

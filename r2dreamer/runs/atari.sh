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
    "atari_alien"
    "atari_amidar"
    "atari_assault"
    "atari_asterix"
    "atari_bank_heist"
    "atari_battle_zone"
    "atari_boxing"
    "atari_breakout"
    "atari_chopper_command"
    "atari_crazy_climber"
    "atari_demon_attack"
    "atari_freeway"
    "atari_frostbite"
    "atari_gopher"
    "atari_hero"
    "atari_jamesbond"
    "atari_kangaroo"
    "atari_krull"
    "atari_kung_fu_master"
    "atari_ms_pacman"
    "atari_pong"
    "atari_private_eye"
    "atari_qbert"
    "atari_road_runner"
    "atari_seaquest"
    "atari_up_n_down"
)

# ==== Loop ====
for task in "${tasks[@]}"
do
    for seed in $(seq $SEED_START $SEED_STEP $SEED_END)
    do
        CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
            env=atari100k \
            env.task=$task \
            logdir=logdir/${DATE}_${METHOD}_${task#atari_}_$seed \
            model.compile=True \
            device=cuda:0 \
            buffer.storage_device=cuda:0 \
            model=size200M \
            model.rep_loss=${METHOD} \
            seed=$seed
    done
done

#!/bin/bash

# ==== Common settings ====
GPU_ID=0
DATE=$(date +%m%d) # auto complete
SEED_START=0
SEED_END=400
SEED_STEP=100
METHOD=r2dreamer

# ==== Run loop ====
for seed in $(seq $SEED_START $SEED_STEP $SEED_END)
do
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
        env=crafter \
        logdir=logdir/${DATE}_crafter_$seed \
        model.compile=True \
        device=cuda:0 \
        buffer.storage_device=cuda:0 \
        model.rep_loss=${METHOD} \
        model=size200M \
        seed=$seed
done

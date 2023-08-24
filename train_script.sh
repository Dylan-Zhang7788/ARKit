#!/usr/bin/env bash

LOG_ALIAS=$1
LOG_DIR="ckpts_basis156_0817/logs"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/`date +'%Y-%m-%d_%H:%M.%S'`.log"

CUDA_VISIBLE_DEVICES=0,1  python main_train.py --arch="mobilenet_v2" \
    --start-epoch=1 \
    --snapshot="ckpts_basis156_0817/SynergyNet" \
    --param-fp-train='./3dmm_data/param_all_norm_v201.pkl' \
    --warmup=5 \
    --batch-size=2048 \
    --base-lr=0.05 \
    --epochs=100 \
    --milestones=48,64 \
    --print-freq=50 \
    --devices-id='0,1' \
    --workers=8 \
    --filelists-train="./3dmm_data/train_data_0809_small.txt" \
    --root="./train_aug_120x120" \
    --log-file="${LOG_FILE}" \
    --test_initial=True \
    --save_val_freq=10 \
    --resume="" \
    --model_path='./ckpts_basis156_0811/SynergyNet_checkpoint_epoch_100.pth.tar' \

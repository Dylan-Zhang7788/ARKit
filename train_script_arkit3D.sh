#!/usr/bin/env bash

LOG_ALIAS=$1
LOG_DIR="ckpts_arkit_0824/logs"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/`date +'%Y-%m-%d_%H:%M.%S'`.log"

python main_train_arkit3D.py --arch="resnet101" \
    --start-epoch=1 \
    --snapshot="ckpts_arkit_0824/SynergyNet" \
    --param-fp-train='./3dmm_data/param_all_norm_v201.pkl' \
    --warmup=5 \
    --batch-size=256 \
    --base-lr=0.08 \
    --epochs=100 \
    --milestones=48,64 \
    --print-freq=50 \
    --devices-id='0,1' \
    --workers=8 \
    --filelists-train="./3dmm_data/train_data_0809_small.txt" \
    --root="/data1/zhang.hongshuang/arkit_data" \
    --log-file="${LOG_FILE}" \
    --test_initial=True \
    --save_val_freq=10 \
    --resume="ckpts_arkit_0823/SynergyNet_checkpoint_epoch_100.pth.tar" \



#!/bin/sh
CUR_DIR=$(cd $(dirname "$0")/..;pwd)
echo "current path: "$CUR_DIR
export PYTHONPATH="${CUR_DIR}":$PYTHONPATH
export DatasetPath="${CUR_DIR}/datasets"

# bert训练
CUDA_VISIBLE_DEVICES=0 python -u runs/run_bert_cls.py \
    --do_train \
    --data_dir ${DatasetPath} \
    --task_name JD \
    --model_type bert \
    --model_name_or_path "./bert-base-chinese" \
    --overwrite_output_dir \
    --output_dir "./outputs" \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --per_gpu_train_batch_size 64 \
    --per_gpu_eval_batch_size 256 \
    --logging_steps 100

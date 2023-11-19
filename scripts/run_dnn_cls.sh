#!/bin/sh
CUR_DIR=$(cd $(dirname "$0")/..;pwd)
echo "current path: "$CUR_DIR
export PYTHONPATH="${CUR_DIR}":$PYTHONPATH
export DatasetPath="${CUR_DIR}/datasets"

# dnn 训练模型
CUDA_VISIBLE_DEVICES=0 python -u runs/run_dnn_cls.py \
    --do_train \
    --data_dir ${DatasetPath} \
    --task_name JD \
    --model_type cnn \
    --loss_type ce \
    --data_format ltw \
    --word_type True \
    --overwrite_output_dir \
    --output_dir "${CUR_DIR}/outputs" \
    --num_train_epochs 10 \
    --learning_rate 2e-4 \
    --per_gpu_train_batch_size 64 \
    --per_gpu_eval_batch_size 128 \
    --logging_steps 100

#!/bin/sh
CUR_DIR=$(cd $(dirname "$0")/..;pwd)
echo "current path: "$CUR_DIR
export PYTHONPATH="${CUR_DIR}":$PYTHONPATH
export DatasetPath="${CUR_DIR}/datasets"

# 传统机器学习
python -u runs/run_ml_cls.py \
    --do_train \
    --data_dir ${DatasetPath} \
    --task_name jd \
    --model_type dt \
    --output_dir "./outputs" \
    --n_jobs 4

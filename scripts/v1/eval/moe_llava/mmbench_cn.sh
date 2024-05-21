#!/bin/bash

SPLIT="mmbench_dev_cn_20231003"
CONV="phi"
CKPT_NAME="llavaphi-2.7b-finetune-moe"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="/home/hanqing/data/eval"

deepspeed --include localhost:$1 --master_port $(($1 + 29500)) moellava/eval/model_vqa_mmbench.py \
    --model-path ${CKPT} \
    --question-file ${EVAL}/mmbench/$SPLIT.tsv \
    --answers-file ${EVAL}/mmbench/answers/$SPLIT/${CKPT_NAME}.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode ${CONV}

mkdir -p ${EVAL}/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ${EVAL}/mmbench/$SPLIT.tsv \
    --result-dir ${EVAL}/mmbench/answers/$SPLIT \
    --upload-dir ${EVAL}/mmbench/answers_upload/$SPLIT \
    --experiment ${CKPT_NAME} > ${EVAL}/mmbench/${CKPT_NAME}-cn.txt
	

#!/bin/bash

CONV="phi"
CKPT_NAME="llavaphi-2.7b-finetune-moe-mousi-linear-grapht"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="/home/hanqing/data/eval"
deepspeed --include localhost:$1 --master_port $(($1 + 29500)) moellava/eval/model_vqa_loader.py \
    --model-path ${CKPT} \
    --question-file ${EVAL}/MME/llava_mme.jsonl \
    --image-folder ${EVAL}/MME/MME_Benchmark_release_version \
    --answers-file ${EVAL}/MME/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

cd ${EVAL}/MME

python convert_answer_to_mme.py --experiment $CKPT_NAME

cd eval_tool

python calculation.py --results_dir answers/$CKPT_NAME > ${EVAL}/MME/${CKPT_NAME}.txt


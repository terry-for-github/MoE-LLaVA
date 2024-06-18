#!/bin/bash


CONV="phi"
CKPT_NAME="llavaphi-2.7b-finetune-moe-mousi-linear-grapht"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="/home/hanqing/data/eval"

deepspeed --include localhost:$1 --master_port $(($1 + 29500)) moellava/eval/model_vqa_loader.py \
    --model-path ${CKPT} \
    --question-file ${EVAL}/pope/llava_pope_test.jsonl \
    --image-folder ${EVAL}/pope/val2014 \
    --answers-file ${EVAL}/pope/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

python3 moellava/eval/eval_pope.py \
    --annotation-dir ${EVAL}/pope/coco \
    --question-file ${EVAL}/pope/llava_pope_test.jsonl \
    --result-file ${EVAL}/pope/answers/${CKPT_NAME}.jsonl > ${EVAL}/pope/${CKPT_NAME}.txt
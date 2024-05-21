#!/bin/bash


CONV="phi"
CKPT_NAME="llavaphi-2.7b-finetune-moe-mousi"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="/home/hanqing/data/eval"
deepspeed --include localhost:$1 --master_port $(($1 + 29500)) moellava/eval/model_vqa_science.py \
    --model-path ${CKPT} \
    --question-file ${EVAL}/scienceqa/llava_test_CQM-A.json \
    --image-folder ${EVAL}/scienceqa/images/test \
    --answers-file ${EVAL}/scienceqa/answers/${CKPT_NAME}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode ${CONV}

python3 moellava/eval/eval_science_qa.py \
    --base-dir ${EVAL}/scienceqa \
    --result-file ${EVAL}/scienceqa/answers/${CKPT_NAME}.jsonl \
    --output-file ${EVAL}/scienceqa/answers/${CKPT_NAME}_output.jsonl \
    --output-result ${EVAL}/scienceqa/answers/${CKPT_NAME}_result.json > ${EVAL}/scienceqa/${CKPT_NAME}.txt
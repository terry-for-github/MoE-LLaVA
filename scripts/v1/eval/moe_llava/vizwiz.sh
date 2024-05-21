#!/bin/bash

CONV="phi"
CKPT_NAME="llavaphi-2.7b-finetune-moe"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="/home/hanqing/data/eval"
deepspeed --include localhost:$1 --master_port $(($1 + 29500)) moellava/eval/model_vqa_loader.py \
    --model-path ${CKPT} \
    --question-file ${EVAL}/vizwiz/llava_test.jsonl \
    --image-folder ${EVAL}/vizwiz/test \
    --answers-file ${EVAL}/vizwiz/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

# I have no openai-apikey
#python3 scripts/convert_vizwiz_for_submission.py \
#    --annotation-file ${EVAL}/vizwiz/llava_test.jsonl \
#    --result-file ${EVAL}/vizwiz/answers/${CKPT_NAME}.jsonl \
#    --result-upload-file ${EVAL}/vizwiz/answers_upload/${CKPT_NAME}.json > ${EVAL}/vizwiz/${CKPT_NAME}.txt
#!/bin/bash


CONV="phi"
CKPT_NAME="llavaphi-2.7b-finetune-moe-mousi"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="/home/hanqing/data/eval"
deepspeed --include localhost:$1 --master_port $(($1 + 29500)) moellava/eval/model_vqa_loader.py \
    --model-path ${CKPT} \
    --question-file ${EVAL}/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ${EVAL}/textvqa/train_images \
    --answers-file ${EVAL}/textvqa/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

python3 -m moellava.eval.eval_textvqa \
    --annotation-file ${EVAL}/textvqa/TextVQA_0.5.1_val.json \
    --result-file ${EVAL}/textvqa/answers/${CKPT_NAME}.jsonl > ${EVAL}/textvqa/${CKPT_NAME}.txt
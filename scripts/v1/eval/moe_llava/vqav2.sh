#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
unset CUDA_VISIBLE_DEVICES
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CONV="phi"
CKPT_NAME="llavaphi-2.7b-finetune-moe"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="/home/hanqing/data/eval"
SPLIT="llava_vqav2_mscoco_test-dev2015"

#for IDX in $(seq 0 $((CHUNKS-1))); do
#    deepspeed --include localhost:${GPULIST[$IDX]} --master_port $((${GPULIST[$IDX]} + 29501)) moellava/eval/model_vqa_loader.py \
#        --model-path ${CKPT} \
#        --question-file ${EVAL}/vqav2/$SPLIT.jsonl \
#        --image-folder ${EVAL}/vqav2/test2015 \
#        --answers-file ${EVAL}/vqav2/answers/$SPLIT/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl \
#        --num-chunks $CHUNKS \
#        --chunk-idx $IDX \
#        --temperature 0 \
#        --conv-mode ${CONV} &
#done

#wait

output_file=${EVAL}/vqav2/answers/$SPLIT/${CKPT_NAME}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${EVAL}/vqav2/answers/$SPLIT/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python3 scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt ${CKPT_NAME} --dir ${EVAL}/vqav2
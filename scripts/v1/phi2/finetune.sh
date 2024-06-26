#!/bin/bash

HOME_FOLDER="/home/hanqing"
CODE_FOLDER="${HOME_FOLDER}/code"
DATA_FOLDER="${HOME_FOLDER}/data"
MODEL_FOLDER="${HOME_FOLDER}/models"
JSON_FOLDER="${DATA_FOLDER}/MoE-LLaVA-Json"
IMAGE_FOLDER="${DATA_FOLDER}/MoE-LLaVA-Image"
CHECKPOINTS_FOLDER="./checkpoints/llavaphi-2.7b-pretrain-mousi-linear-test"

cd ${CODE_FOLDER}/MoE-LLaVA
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed moellava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${MODEL_FOLDER}/microsoft_phi-2 \
    --version phi \
    --data_path ${JSON_FOLDER}/la_tune_256k.json \
                ${JSON_FOLDER}/lrv_tune_331k.json ${JSON_FOLDER}/lvis_tune_220k_.json \
                ${JSON_FOLDER}/svit_tune_157k.json ${JSON_FOLDER}/nlp_tune.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower ${MODEL_FOLDER}/openai_clip-vit-large-patch14-336 \
    --image_projector_type linear \
    --pretrain_mm_mlp_adapter ${CHECKPOINTS_FOLDER}/mm_projector.bin \
    --pretrain_dino_mm_mlp_adapter ${CHECKPOINTS_FOLDER}/dino_mm_projector.bin \
    --pretrain_ocr_mm_mlp_adapter ${CHECKPOINTS_FOLDER}/ocr_mm_projector.bin \
    --pretrain_fusion_mm_mlp_adapter ${CHECKPOINTS_FOLDER}/fusion_mm_projector.bin \
    --pretrain_graph_mm_mlp_adapter ${CHECKPOINTS_FOLDER}/graph_mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llavaphi-2.7b-finetune-mousi-linear-test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"


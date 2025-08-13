#!/bin/bash
#SBATCH --job-name=blip_training # Job name
#SBATCH --partition=GPU-shared # Partition (queue) name
#SBATCH -N 1
#SBATCH --gres=gpu:v100-32:4 # Request 2 A100 GPU
#SBATCH --time=96:00:00                      # Time limit (hh:mm:ss)
#SBATCH --output=blip_training.out # Standard output
#SBATCH --error=blip_training.err # Standard error


export HF_HOME=../training_blip/hf_home/
export OUTPUT_FOLDER=../training_blip/output_folder/
export IMG_FOLDER=../training_blip/img_folder/


srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$HOSTNAME:29501 blip3o/train/train_mem.py \
    --deepspeed ./scripts/zero1.json \
    #--model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct  \
    --model_name_or_path ../../hf_cache/hub/models--BLIP3o--BLIP3o-Model-8B/snapshots/c2edfc20814d4624c8d73ca3de351ebc3fa86508/ \
    --version qwen \
    --data_type "mix" \
    --image_folder ${IMG_FOLDER} \
    --gen_vision_tower eva-clip-E-14-plus \
    --gen_projector_type mlp2x_gelu \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_FOLDER} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.003 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr":1e-5}' \
    --model_max_length 512 \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --gen_pooling early_pool2d_4 \
    --n_query 64 \
    --n_und_query 0 \
    --report_to wandb \
    --run_name blip3o_qwen_vl_7b

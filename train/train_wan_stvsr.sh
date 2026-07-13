#!/usr/bin/env bash
set -euo pipefail
# bash train/train_wan_stvsr.sh
export CUDA_VISIBLE_DEVICES=5,6

PROJECT_ROOT=/data2/wujialing/project/STVSR/WanSTVSR
cd ${PROJECT_ROOT}

python train/train_wan_stvsr.py \
  --gpu_num 2 \
  --training_strategy deepspeed_stage_2 \
  --output_path ./checkpoint/wanstvsr_baseline \
  --train_data_root data/train/HQ-VSR \
  --val_lq_root data/test/UDM10/LQ-Video-temporal \
  --val_hq_root data/test/UDM10/GT-Video\
  --degradation_config dataset/train_degradation.yaml \
  --dit_path /data2/wujialing/pretrained_weights/DiffSynth-Studio/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors \
  --vae_path /data2/wujialing/pretrained_weights/DiffSynth-Studio/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.safetensors \
  --empty_prompt_embedding_path ./checkpoint/empty_prompt_embedding.pth \
  --num_frames 9 \
  --height 240 \
  --width 480 \
  --inter_frame_margin 6 \
  --batch_size_per_gpu 1 \
  --accumulate_grad_batches 4 \
  --max_steps 10000 \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --learning_rate 5.0e-5 \
  --weight_decay 1.0e-4 \
  --lr_scheduler constant_with_warmup \
  --warmup_steps 500 \
  --val_check_interval 500 \
  --fixed_timestep 799 \
  --noise_step 0 \
  --use_consis_loss \
  --lambda_consis 1.0 \
  --metrics psnr,ssim,lpips,dists,clipiqa \
  --tensorboard_name WanSTVSR-baseline-v1-single_stage \
  2>&1 | tee logs/train.log

# tail -f logs/train.log

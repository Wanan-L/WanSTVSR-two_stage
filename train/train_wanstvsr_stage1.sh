# !/usr/bin/env bash
set -euo pipefail
# bash train/train_wanstvsr_stage1.sh
export CUDA_VISIBLE_DEVICES=3,5

PROJECT_ROOT=/data2/wujialing/project/STVSR/WanSTVSR-0713
cd ${PROJECT_ROOT}

python train/train_wanstvsr_stage1.py \
  --gpu_num 2 \
  --training_strategy deepspeed_stage_2 \
  --output_path ./checkpoint/wanstvsr_baseline-v2-two_stage1 \
  --train_data_root data/train/HQ-VSR \
  --val_lq_root data/test/UDM10/LQ-Video-temporal \
  --val_hq_root data/test/UDM10/GT-Video \
  --degradation_config dataset/degradation_video.yaml \
  --dit_path /data2/wujialing/pretrained_weights/DiffSynth-Studio/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors \
  --vae_path /data2/wujialing/pretrained_weights/DiffSynth-Studio/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.safetensors \
  --empty_prompt_embedding_path checkpoint/empty_prompt_embedding.pth \
  --resume_from_checkpoint checkpoint/wanstvsr_baseline-v2-two_stage1/tensorboardlog/WanSTVSR-baseline-v2-two_stage1/version_0/checkpoints/epoch=39-step=10000.ckpt \
  --num_frames 33 \
  --height 320 \
  --width 640 \
  --inter_frame_margin 8 \
  --batch_size_per_gpu 4 \
  --accumulate_grad_batches 1 \
  --max_steps 21000 \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --learning_rate 2.0e-5 \
  --weight_decay 1.0e-4 \
  --lr_scheduler constant_with_warmup \
  --warmup_steps 500 \
  --val_check_interval 500 \
  --fixed_timestep 799 \
  --noise_step 0 \
  --metrics psnr,ssim,lpips,dists,clipiqa \
  --tensorboard_name WanSTVSR-baseline-v2-two_stage1 \
  2>&1 | tee train_logs/WanSTVSR-baseline-v2-two_stage1.log

# tail -f logs/train.log
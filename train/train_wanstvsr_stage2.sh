# !/usr/bin/env bash
set -euo pipefail
# bash train/train_wanstvsr_stage2.sh
export CUDA_VISIBLE_DEVICES=0,5

PROJECT_ROOT=/data2/wujialing/project/STVSR/WanSTVSR-0713
cd ${PROJECT_ROOT}

python train/train_wanstvsr_stage2.py \
  --gpu_num 2 \
  --training_strategy deepspeed_stage_2 \
  --output_path ./checkpoint/wanstvsr_baseline-v2-two_stage2-init_from_stage1_18000 \
  --video_data_root data/train/HQ-VSR \
  --image_data_root /data2/wujialing/data/ISR/DIV2K/DIV2K_train_HR \
  --val_lq_root data/test/UDM10/LQ-Video-temporal \
  --val_hq_root data/test/UDM10/GT-Video \
  --degradation_config dataset/degradation_image_video.yaml \
  --init_from_checkpoint checkpoint/wanstvsr_baseline-v2-two_stage1/tensorboardlog/WanSTVSR-baseline-v2-two_stage1/version_1/checkpoints/epoch=70-step=18000.ckpt/output_dir \
  --dit_path /data2/wujialing/pretrained_weights/DiffSynth-Studio/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors \
  --vae_path /data2/wujialing/pretrained_weights/DiffSynth-Studio/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.safetensors \
  --empty_prompt_embedding_path ./checkpoint/empty_prompt_embedding.pth \
  --image_ratio 0.5 \
  --num_frames 9 \
  --height 320 \
  --width 320 \
  --inter_frame_margin 5 \
  --batch_size_per_gpu 1 \
  --accumulate_grad_batches 8 \
  --max_steps 11000 \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --learning_rate 5.0e-6 \
  --warmup_steps 500 \
  --val_check_interval 500 \
  --fixed_timestep 799 \
  --noise_step 0 \
  --metrics psnr,ssim,lpips,dists,clipiqa \
  --tensorboard_name WanSTVSR-baseline-v2-two_stage2-init_from_stage1_18000 \
  2>&1 | tee train_logs/WanSTVSR-baseline-v2-two_stage2-init_from_stage1_18000.log

# tail -f logs/train.log
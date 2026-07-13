CUDA_VISIBLE_DEVICES="0" python ./dataset/data_preparation.py \
  --data_root ./data/train/HQ-VSR\
  --degradation_config ./dataset/train_degradation.yaml \
  --save_path ./dataset/pth_file \
  --output_path ./data_logs \
  --num_frames 5 \
  --height 240 \
  --width 480 \
  --spatial_scale 4 \
  --temporal_scale 2 \
  --inter_frame_margin 5 \
  --text_encoder_path /data2/wujialing/pretrained_weights/DiffSynth-Studio/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.safetensors \
  --vae_path /data2/wujialing/pretrained_weights/DiffSynth-Studio/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.safetensors

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8" python ./dataset/data_preparation.py \
  --video_path ./dataset/video_list.txt \
  --caption_path ./dataset/caption \
  --save_path ./dataset/pth_file \
  --output_path ./data_logs \
  --text_encoder_path ./model_checkpoints/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth \
  --vae_path ./model_checkpoints/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
cd /data2/wujialing/project/STVSR/WanSTVSR
CUDA_VISIBLE_DEVICES=5 python inference/test_wan_stvsr.py \
  --wan_model_path /data2/wujialing/pretrained_weights/DiffSynth-Studio/Wan-AI/Wan2.1-T2V-1.3B \
  --wanstvsr_model_path ./checkpoint/wanstvsr_baseline/converted_fp32_epoch27_step7000/pytorch_model.bin \
  --empty_prompt_embedding_path ./checkpoint/empty_prompt_embedding.pth \
  --video_path /data2/wujialing/data/VSR/UDM10/LQ-Video-temporal \
  --save_path ./results/UDM10 \
  --spatial_scale 4 \
  --temporal_scale 2 
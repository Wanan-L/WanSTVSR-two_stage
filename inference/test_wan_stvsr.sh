cd /data2/wujialing/project/STVSR/WanSTVSR-0713
CUDA_VISIBLE_DEVICES=2 python inference/test_wan_stvsr.py \
  --wan_model_path /data2/wujialing/pretrained_weights/DiffSynth-Studio/Wan-AI/Wan2.1-T2V-1.3B \
  --wanstvsr_model_path checkpoint/wanstvsr_baseline-v2-two_stage2-init_from_stage1_18000/tensorboardlog/WanSTVSR-baseline-v2-two_stage2-init_from_stage1_18000/version_0/checkpoints/epoch=31-step=4000.ckpt/output_dir \
  --video_path /data2/wujialing/data/VSR/UDM10/LQ-Video-temporal \
  --save_path results_test/UDM10 \
  --spatial_scale 4 \
  --temporal_scale 2 \
  --prompt  "A clear, natural, high-quality restored video with faithful scene structure, realistic fine details, stable edges, consistent textures across frames, natural colors, smooth motion, and strong temporal consistency." \
  --negative_prompt "blur, flickering, temporal inconsistency, unstable details, hallucinated textures, oversharpening, ringing artifacts, ghosting, warped structures, color shifts, noise, compression artifacts, excessive smoothing, low quality" \
  --cfg_scale 5.0
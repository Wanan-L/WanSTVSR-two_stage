#!/bin/bash
# conda activate torch2.8
# cd /data2/wujialing/project/STVSR/WanSTVSR
# bash inference/batch_test_wan_stvsr.sh

# List of datasets
DATASETS_X1=()
DATASETS_X4=("UDM10")
# "UDM10" "SPMCS" "YouHQ40" "REDS"
# "RealVSR" "MVSR4x"

GPU_ID="5"
INPUT_ROOT="/data2/wujialing/data/VSR"

mkdir -p "logs"

run_inference() {
    local dataset="$1"
    local upscale="$2"

    echo "Running WanSTVSR inference on dataset: ${dataset}, upscale: ${upscale}"

    CUDA_VISIBLE_DEVICES="$GPU_ID" python -m inference.test_wan_stvsr \
    --wan_model_path /data2/wujialing/pretrained_weights/DiffSynth-Studio/Wan-AI/Wan2.1-T2V-1.3B \
    --wanstvsr_model_path ./checkpoint/wanstvsr_baseline/converted_fp32_epoch27_step7000/pytorch_model.bin \
    --empty_prompt_embedding_path ./checkpoint/empty_prompt_embedding.pth \
    --video_path "${INPUT_ROOT}/${dataset}/LQ-Video-temporal" \
    --save_path "results/${dataset}" \
    --spatial_scale "$upscale" \
    --temporal_scale 2 > "logs/${dataset}.log" 2>&1
}

for dataset in "${DATASETS_X4[@]}"; do
    run_inference "$dataset" 4
done

for dataset in "${DATASETS_X1[@]}"; do
    run_inference "$dataset" 1
done
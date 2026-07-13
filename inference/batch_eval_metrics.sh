#!/bin/bash
# conda activate torch2.8
# cd /data2/wujialing/project/STVSR/WanSTVSR
# bash inference/batch_eval_metrics.sh

set -e

# List of datasets
DATASETS_X1=("MVSR4x")
DATASETS_X4=("UDM10" "SPMCS")
# "UDM10" "SPMCS" "YouHQ40" "REDS"
# "RealVSR" "MVSR4x"

GPU_ID="4"
INPUT_ROOT="/data2/wujialing/data/VSR"
RESULT_ROOT="results"

METRICS="psnr,ssim,lpips,dists,clipiqa,clipiqa+,niqe,ilniqe,liqe,musiq,maniqa,brisque,dover,ewarp,vbench,fastvqa,mdvqa"

mkdir -p "logs"

run_eval_metrics() {
    local dataset="$1"

    local pred_dir="${RESULT_ROOT}/${dataset}"
    local gt_dir="${INPUT_ROOT}/${dataset}/GT-Video"
    local out_dir="${RESULT_ROOT}/${dataset}"

    echo "Evaluating WanVSR metrics on dataset: ${dataset}"
    echo "  pred: ${pred_dir}"
    echo "  gt:   ${gt_dir}"
    echo "  out:  ${out_dir}"

    CUDA_VISIBLE_DEVICES="$GPU_ID" python -m inference.eval_metrics \
    --pred "$pred_dir" \
    --gt "$gt_dir" \
    --out "$out_dir" \
    --metrics "$METRICS" \
    --filename all_metrics_results.json > "logs/eval_metrics_${dataset}.log" 2>&1
}

for dataset in "${DATASETS_X4[@]}"; do
    run_eval_metrics "$dataset"
done

for dataset in "${DATASETS_X1[@]}"; do
    run_eval_metrics "$dataset"
done

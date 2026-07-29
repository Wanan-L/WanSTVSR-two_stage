cd /data2/wujialing/project/STVSR/WanSTVSR-0713
CUDA_VISIBLE_DEVICES=2 python inference/eval_metrics.py \
    --pred results_test/UDM10 \
    --gt /data2/wujialing/data/VSR/UDM10/GT-Video \
    --out results_test/UDM10 \
    --metrics "psnr,ssim,lpips,dists,clipiqa,clipiqa+,niqe,ilniqe,liqe,musiq,maniqa,brisque,dover,ewarp,vbench,fastvqa,mdvqa" \
    --filename all_metrics_results.json
# WanVSR

WanVSR tests `Wan2.1-T2V-1.3B` as a single-step diffusion baseline for spatio-temporal video super-resolution.

Core setting:
- HQ-VSR root: `/data2/wujialing/data/VSR/HQ-VSR`
- Train crop: 17 frames x 320 x 640
- Spatial degradation: RealBasicVSR-style on-the-fly 4x downsampling
- Temporal degradation: uniformly drop one frame between every two frames
- Alignment: bilinear 4x spatial upsampling + single-frame temporal interpolation to form `LQ_up`
- Fixed diffusion timestep: `t=799`
- Frozen VAE/Text Encoder, train DiT only
- Loss: latent MSE + pixel MSE + LPIPS + 0.1 * bidirectional RAFT consistency

## Project tree

```text
WanVSR/
├── configs/
│   └── wanvsr_1.3b.yaml
├── diffsynth/
├── metrics/                 # copied from SparkVSR when available
├── scripts/
│   ├── train.sh
│   ├── infer.sh
│   └── eval.sh
├── wanvsr/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── hqvsr_dataset.py
│   │   └── degradations.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── wan_vsr.py
│   │   └── flow.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── checkpoint.py
│   │   ├── distributed.py
│   │   ├── video.py
│   │   └── metrics.py
│   ├── train.py
│   ├── infer.py
│   └── eval.py
└── README.md
```

## Notes

`bf16` is enabled by default because Wan2.1 DiT/VAE inference is memory-heavy and bf16 is usually more stable than fp16 on Ampere/Hopper GPUs. Gradient checkpointing is enabled by default and can be disabled in the yaml. The copied DiffSynth code already includes VRAM management utilities and Wan VAE tiling/chunking hooks; this project exposes conservative config switches instead of adding new model modules.

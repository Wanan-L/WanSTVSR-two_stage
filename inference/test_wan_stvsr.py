#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WanSTVSR inference script.

This script uses the WanSTVSR pipeline and the same input/output tensor utilities as train_wan_stvsr.py validation.

Expected input to WanSTVSRPipeline.test(): [B, C, T, H, W] in [-1, 1].
The helper prepare_input_tensor() should spatially/temporally align the LQ
video according to --spatial_scale and --temporal_scale before inference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm import tqdm

from diffsynth import ModelManager
from diffsynth.pipelines.wan_stvsr import WanSTVSRPipeline
from dataset.utils import prepare_input_tensor, scan_video_or_frame_dirs, save_video

from utils.checkpoint_utils import load_checkpoint_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WanSTVSR inference script")

    # Model paths. 
    parser.add_argument('--wan_model_path', type=str, default='./model_checkpoints/Wan2.1-T2V-1.3B')
    parser.add_argument('--wanstvsr_model_path', type=str, required=True)
    parser.add_argument(
        '--empty_prompt_embedding_path', type=str, default=None,
        help=(
            "Optional precomputed positive embedding. When provided, --prompt and per-video captions are ignored,"
            "but --negative_prompt can still be encoded when CFG is enabled."
        ),
    )

    # Input / output.
    parser.add_argument('--video_path', type=str, default='./VideoLQ/lq')
    parser.add_argument('--caption_path', type=str, default=None, help="Optional json file or json directory. Matched by video stem.")
    parser.add_argument('--save_path', type=str, default='./VideoLQ/results')

    # WanSTVSR one-step settings. Spatio-temporal SR preprocessing. Keep these defaults aligned with training.
    parser.add_argument("--spatial_scale", type=int, default=4)
    parser.add_argument("--temporal_scale", type=int, default=2)
    parser.add_argument('--fixed_timestep', type=int, default=799)
    parser.add_argument('--noise_step', type=int, default=0)
    parser.add_argument('--num_inference_steps', type=int, default=1)
    parser.add_argument('--cfg_scale', type=float, default=1.0)

    # VAE tiling.
    parser.add_argument("--tiled", action="store_true", default=False)
    parser.add_argument("--tile_size_height", type=int, default=34)
    parser.add_argument("--tile_size_width", type=int, default=34)
    parser.add_argument("--tile_stride_height", type=int, default=18)
    parser.add_argument("--tile_stride_width", type=int, default=16)

    # Prompt options. Set cfg_scale != 1 to enable negative-prompt CFG.
    parser.add_argument(
        "--prompt", type=str, default="",
        help="Positive text prompt. Ignored when --empty_prompt_embedding_path is set.",
    )
    parser.add_argument(
        "--negative_prompt", type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，画面，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，畸形的，杂乱的背景",
    )   # cfg !=1 时才会使用

    # Runtime / saving.
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument('--fps', type=int, default=16)
    parser.add_argument('--video_quality', type=int, default=9)
    parser.add_argument('--color_fix', action="store_true", default=False)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

def load_dit_checkpoint(pipe, ckpt_path):
    state_dict = load_checkpoint_state_dict(ckpt_path)
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint is not a state_dict-like object: {ckpt_path}")

    replace_state_dict = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        k = k.replace('module.pipe.dit.', '', 1)
        k = k.replace('pipe.dit.', '', 1)
        k = k.replace('module.dit.', '', 1)
        k = k.replace('dit.', '', 1)
        replace_state_dict[k] = v.detach().cpu()

    if len(replace_state_dict) == 0:
        raise RuntimeError(f"No tensor weights found in checkpoint: {ckpt_path}")

    missing, unexpected = pipe.denoising_model().load_state_dict(replace_state_dict, strict=False)
    print(f'Loaded checkpoint: {ckpt_path}')
    print(f'missing keys: {len(missing)}, unexpected keys: {len(unexpected)}')


def load_prompt_embedding(path, device, dtype):
    if path is None:
        return None

    data = torch.load(path, map_location='cpu', weights_only=False)
    emb = data.get('prompt_emb', data) if isinstance(data, dict) else data
    if isinstance(emb, dict):
        emb = emb['context']
    if emb.dim() == 2:
        emb = emb.unsqueeze(0)
    return {'context': emb.to(device=device, dtype=dtype)}


def load_caption(caption_path, name, default_prompt):
    if caption_path is None:
        return default_prompt

    caption_root = Path(caption_path)
    caption_file = caption_root if caption_root.is_file() else caption_root / f'{name}.json'
    if not caption_file.exists():
        return default_prompt

    with open(caption_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get('caption', data.get('prompt', default_prompt))
    if isinstance(data, str):
        return data
    return default_prompt


def collect_inputs(video_path):
    path = Path(video_path)
    if path.is_file():
        return [path]
    return [Path(p) for p in scan_video_or_frame_dirs(path)]


def get_torch_dtype(dtype: str):
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")

def main() -> None:
    args = parse_args()
    torch_dtype = get_torch_dtype(args.dtype)
    if args.cfg_scale < 0:
        raise ValueError(f"--cfg_scale must be non-negative, got {args.cfg_scale}.")

    model_paths = [
        os.path.join(args.wan_model_path, 'diffusion_pytorch_model.safetensors'),
        os.path.join(args.wan_model_path, "Wan2.1_VAE.safetensors")
    ]
    # Positive prompt text needs T5 when no embedding is supplied.
    # Negative prompt text also needs T5 whenever CFG is enabled.
    if args.empty_prompt_embedding_path is None or args.cfg_scale != 1.0:
        model_paths.append(os.path.join(args.wan_model_path, 'models_t5_umt5-xxl-enc-bf16.safetensors'))

    model_manager = ModelManager(torch_dtype=torch_dtype, device='cpu')
    model_manager.load_models(model_paths)

    pipe = WanSTVSRPipeline.from_model_manager(model_manager, torch_dtype=torch_dtype, device=args.device)
    load_dit_checkpoint(pipe, args.wanstvsr_model_path)
    pipe.requires_grad_(False)
    pipe.eval()
    pipe.dit.to(device=args.device, dtype=torch_dtype).eval()
    pipe.vae.to(device=args.device, dtype=torch_dtype).eval()
    if pipe.text_encoder is not None:
        pipe.text_encoder.to(device=args.device, dtype=torch_dtype).eval()
    # pipe.denoising_model().to(dtype=torch_dtype).eval()
    # pipe.enable_vram_management()   # dit, vae

    prompt_emb_posi = load_prompt_embedding(args.empty_prompt_embedding_path, args.device, torch_dtype)

    video_files = collect_inputs(args.video_path)
    os.makedirs(args.save_path, exist_ok=True)

    for video_file in tqdm(video_files):
        print(f'Processing: {video_file}', flush=True)  # flush 打印后立刻把输出缓冲区刷新到终端或日志中
        prompt = load_caption(args.caption_path, video_file.stem, args.prompt)

        positive_source = (f"embedding: {args.empty_prompt_embedding_path}" if prompt_emb_posi is not None else f"text: {prompt!r}")
        negative_source = (repr(args.negative_prompt) if args.cfg_scale != 1.0 else "disabled (cfg_scale=1)")
        print(f"[Prompt] positive={positive_source}, negative={negative_source}, cfg_scale={args.cfg_scale}", flush=True)

        lq_video, input_fps = prepare_input_tensor(
            str(video_file),
            dtype=torch_dtype,
            device=args.device,
            spatial_scale=args.spatial_scale,
            temporal_scale=args.temporal_scale,
        )
        # print("[After prepare_input_tensor]", lq_video.shape)

        print(
            f"[Preprocessed LQ] shape={tuple(lq_video.shape)}, dtype={lq_video.dtype}, "
            f"range=({lq_video.min().item():.4f}, {lq_video.max().item():.4f})",
            flush=True
        )

        preprocess_save_dir = os.path.join(args.save_path, "preprocessed_lq")
        preprocess_save_path = os.path.join(preprocess_save_dir,f"{video_file.stem}.mp4",)

        # save_video(lq_video, preprocess_save_path, fps=input_fps * args.temporal_scale)

        with torch.no_grad():
            pred_video = pipe.test(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                lq_video=lq_video,
                prompt_emb_posi=prompt_emb_posi,
                cfg_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
                fixed_timestep=args.fixed_timestep,
                noise_step=args.noise_step,
                color_fix=args.color_fix,
                tiled=args.tiled,
                tile_size=(args.tile_size_height, args.tile_size_width),
                tile_stride=(args.tile_stride_height, args.tile_stride_width),
            ).clamp(-1, 1)

        # print("[After pipe.test]", pred_video.shape)
        output_filename = video_file.stem + '.mp4'
        save_video(pred_video, os.path.join(args.save_path, output_filename), fps=args.fps)
        # save_video(frames, os.path.join(args.save_path, output_filename), fps=args.fps, quality=args.video_quality)


if __name__ == '__main__':
    main()


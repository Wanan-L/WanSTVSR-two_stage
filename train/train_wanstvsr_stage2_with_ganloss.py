#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-2 TE training for WanSTVSR with GANLoss.

TE initializes DiT from a TC checkpoint and optimizes pixel L1, DISTS,
high-frequency optical-flow warp consistency, MUSIQ quality, and adversarial
detail losses. The original paired-video validation path is retained for all
requested metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightning as pl
import pyiqa
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from diffsynth import ModelManager
from diffsynth.pipelines.wan_stvsr import WanSTVSRPipeline
from dataset.spatio_temporal_real_sr_image_video_dataset import SpatioTemporalRealSRImageVideoDataset

from dataset.utils import (
    bcthw_to_fchw_01,
    prepare_input_tensor,
    read_video_as_tensor,
    save_video,
    scan_video_or_frame_dirs,
)
from utils.RAFT.raft_bi import RAFT_bi
from utils.memory_utils import (
    free_memory,
    get_memory_statistics,
    reset_peak_memory_stats,
)
from utils.metric_utils import evaluate_video_metrics
from utils.optical_flow_utils import fbConsistencyCheck, flow_warp
from utils.checkpoint_utils import load_checkpoint_state_dict

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stdout)
py_logger = logging.getLogger(__name__)

STAGE_NAME = "stage2-te-gan"

# Run check_raft_flow_direction.py once in the actual project environment, then
# fix this constant to the output that aligns frame j to frame j+1.  
# "backward" is retained here only as the mapping used by the current draft implementation.


class PatchDiscriminator(nn.Module):
    """
    Small frame PatchGAN discriminator for local texture supervision.

    输出一个二维真假分数图，而不是是为整张图像输出一个真假概率
    例如：[B, 3, 320, 640] -> [B, 1, 38, 78]。每个位置判断一个局部区域是否像真实纹理。
    """

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()

        def block(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1),
                nn.GroupNorm(min(32, out_channels), out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.net = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            block(base_channels, base_channels * 2, stride=2),
            block(base_channels * 2, base_channels * 4, stride=2),
            block(base_channels * 4, base_channels * 8, stride=1),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.net(frames)


def checkpoint_stage(checkpoint: Dict[str, Any]) -> Optional[str]:
    stage = checkpoint.get("wan_stvsr_stage")
    if stage is not None:
        return str(stage)
    source_args = checkpoint.get("wan_stvsr_args", {})
    if isinstance(source_args, dict):
        value = source_args.get("train_stage")
        return None if value is None else str(value)
    return None


class PairedValidationVideoDataset(Dataset):
    """Pair validation LQ/HQ videos or frame directories by stem."""
    def __init__(self, lq_root, hq_root, max_samples: Optional[int] = None):
        super().__init__()
        self.lq_root = Path(lq_root)
        self.hq_root = Path(hq_root)

        lq_samples = scan_video_or_frame_dirs(self.lq_root)
        hq_samples = scan_video_or_frame_dirs(self.hq_root)
        hq_by_stem = {path.stem: path for path in hq_samples}

        pairs: List[Tuple[Path, Path]] = []
        for lq_path in lq_samples:
            hq_path = hq_by_stem.get(lq_path.stem)
            if hq_path is not None:
                pairs.append((lq_path, hq_path))

        if max_samples is not None and max_samples > 0:
            pairs = pairs[:max_samples]
        if not pairs:
            raise FileNotFoundError(f"No paired validation samples found in {self.lq_root} and {self.hq_root}")
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, str]:
        lq_path, hq_path = self.pairs[index]
        return {"lq_path": str(lq_path), "hq_path": str(hq_path), "name": lq_path.stem}


class LightningModelForStage2(pl.LightningModule):
    strict_loading = False

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.args = args
        self.torch_dtype = torch.bfloat16
        self.tiler_kwargs = {
            "tiled": args.tiled,
            "tile_size": (args.tile_size_height, args.tile_size_width),
            "tile_stride": (args.tile_stride_height, args.tile_stride_width),
        }
        self._shape_logged = False

        # Load models
        model_paths = [args.dit_path, args.vae_path]
        # Only load text encoder when empty prompt embedding is not provided.
        if not args.empty_prompt_embedding_path and args.text_encoder_path:
            model_paths.append(args.text_encoder_path)

        model_manager = ModelManager(torch_dtype=self.torch_dtype, device="cpu")
        model_manager.load_models(model_paths)
        self.pipe = WanSTVSRPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()

        self._init_checkpoint_loaded = False
        if args.init_from_checkpoint:
            self._load_initial_dit(args.init_from_checkpoint)

        self.prompt_text = args.prompt
        self.context: Optional[torch.Tensor] = None     # 使用统一的 context
        self.dists_loss: Optional[Any] = None
        self.musiq: Optional[Any] = None                # nqa 正则化
        self.raft: Optional[Any] = None
        self.discriminator = PatchDiscriminator(base_channels=64) # 初始化判别器
        self.metric_models: Dict[str, Any] = {}
        self._musiq_gradient_checked = False
        self._last_memory_log_step = -1         # 避免梯度累积时，同一个 global_step 被 on_train_batch_end 重复记录

        coordinates = torch.arange(5, dtype=torch.float32) - 2.0
        gaussian_1d = torch.exp(-(coordinates**2) / 2.0)
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        self.register_buffer("gaussian_kernel_5x5", gaussian_2d.view(1, 1, 5, 5), persistent=False)

    def sample_image_batch(self) -> bool:
        """让所有分布式 rank 使用相同的图像/视频选择结果。"""
        decision = torch.zeros(1, device=self.device, dtype=torch.int64,)

        if not dist.is_available() or not dist.is_initialized():
            return random.random() < self.args.image_ratio

        if dist.get_rank() == 0:
            decision[0] = int(random.random() < self.args.image_ratio)

        dist.broadcast(decision, src=0)
        return bool(decision.item())

    def freeze_parameters(self) -> None:
        # Freeze the full pipeline first, then enable the trainable part.
        self.pipe.requires_grad_(False)
        self.pipe.eval()

        self.pipe.denoising_model().requires_grad_(True)
        self.pipe.denoising_model().train()

        # Keep trainable DiT params fp32 for optimizer stability.
        for parameter in self.pipe.dit.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()

    def _load_initial_dit(self, checkpoint_path: str) -> None:
        """
        从转换后的分片权重目录中加载 DiT 权重，不恢复 optimizer、scheduler 和 global_step。

        GAN 第二阶段允许从以下 checkpoint 初始化：
        1. stage1-tc：直接从第一阶段开始 GAN 第二阶段训练；
        2. stage2-te：从已经完成的普通第二阶段继续做 GAN 微调。
        """
        checkpoint_dir = Path(checkpoint_path)

        if not checkpoint_dir.is_dir():
            raise NotADirectoryError(f"Expected a converted checkpoint directory, got: {checkpoint_path}")

        # 转换后的 output_dir 通常只保存模型权重，不包含阶段元数据。
        state_dict = load_checkpoint_state_dict(checkpoint_dir)

        if not isinstance(state_dict, dict):
            raise TypeError(f"Unsupported checkpoint state_dict type: {type(state_dict)}")

        model_state = self.pipe.dit.state_dict()
        dit_state: Dict[str, torch.Tensor] = {}

        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            
            if key.startswith("pipe.dit."):
                target_key = key[len("pipe.dit."):]
            elif key.startswith("dit."):
                target_key = key[len("dit."):]
            else:
                target_key = key

            if target_key not in model_state:
                continue

            if value.shape != model_state[target_key].shape:
                raise RuntimeError(
                    f"Shape mismatch for {target_key}: "
                    f"checkpoint={tuple(value.shape)}, "
                    f"model={tuple(model_state[target_key].shape)}"
                )

            dit_state[target_key] = value

        if not dit_state:
            example_keys = list(state_dict.keys())[:10]
            raise ValueError( f"No matching DiT parameters were found in {checkpoint_path}. First checkpoint keys: {example_keys}")

        missing_keys = sorted(set(model_state.keys()) - set(dit_state.keys()))
        if missing_keys:
            raise RuntimeError(
                "Stage1/Stage2-te checkpoint does not contain all DiT parameters. "
                f"loaded={len(dit_state)}, "
                f"model_total={len(model_state)}, "
                f"missing={len(missing_keys)}, "
                f"first_missing={missing_keys[:10]}"
            )

        self.pipe.dit.load_state_dict(dit_state, strict=True)
        self._init_checkpoint_loaded = True
        py_logger.info(
            "Successfully initialized Stage2-GAN DiT: loaded=%d, model_total=%d, path=%s",
            len(dit_state), len(model_state), checkpoint_path,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.context is None:
            self.context = self._load_or_encode_context()
        if stage in ("fit", None):
            self._init_losses()
        # if stage in ("fit", "validate", "test", None):
        #     self._init_metrics()

    def _init_losses(self) -> None:
        if self.dists_loss is None:
            self.dists_loss = pyiqa.create_metric("dists", device=self.device, as_loss=True).eval()
            self.dists_loss.requires_grad_(False)

        if self.musiq is None:
            self.musiq = pyiqa.create_metric("musiq", device=self.device, as_loss=True).eval()
            self.musiq.requires_grad_(False)

        if self.raft is None:
            self.raft = RAFT_bi(self.args.raft_ckpt_path, device=self.device)
            self.raft.requires_grad_(False)
            self.raft.eval()

    def _init_metrics(self) -> None:
        """Create validation metrics separately from TE training losses."""
        if self.metric_models:
            return
        
        metric_names = [name.strip().lower() for name in self.args.metrics.split(",") if name.strip()]
        for name in metric_names:
            if name in self.metric_models:
                continue
            metric = pyiqa.create_metric(name, device=self.device).eval()
            for parameter in metric.parameters():
                parameter.requires_grad_(False)
            self.metric_models[name] = metric
        
        py_logger.info("Initialized validation metrics on %s: %s", self.device, ", ".join(metric_names))

    def _load_or_encode_context(self) -> torch.Tensor:
        if self.args.empty_prompt_embedding_path:
            # Load empty prompt embedding，兼容多种格式
            data = torch.load(self.args.empty_prompt_embedding_path, map_location="cpu", weights_only=False)    # 文件本身就是 tensor
            embedding = data.get("prompt_emb", data) if isinstance(data, dict) else data    # 文件是 dict，里面有 prompt_emb
            if isinstance(embedding, dict):             # 文件是 dict，prompt_emb 里面又是 dict
                embedding = embedding["context"]
            if not isinstance(embedding, torch.Tensor):
                raise TypeError(
                    "The prompt embedding file must contain a Tensor or a dict "
                    "containing prompt_emb/context."
                )
            return embedding.detach().cpu()

        # 没有 embedding 文件，使用 text encoder 编码 prompt
        if self.pipe.text_encoder is None:
            raise ValueError("Provide --empty_prompt_embedding_path or --text_encoder_path")

        old_device = self.pipe.device
        self.pipe.device = self.device
        self.pipe.text_encoder.to(self.device, dtype=self.torch_dtype).eval()
        with torch.no_grad():
            embedding = self.pipe.encode_prompt(self.prompt_text, positive=True)["context"].detach().cpu()
        self.pipe.text_encoder.to("cpu")
        self.pipe.device = old_device
        free_memory()
        return embedding

    def get_context(self, batch_size: int) -> torch.Tensor:
        """处理编码好的 context embedding 到 DiT forward 需要的形状 [B, L, C]"""
        if self.context is None:
            self.context = self._load_or_encode_context()

        context = self.context.to(device=self.device, dtype=self.torch_dtype)
        if context.dim() == 2:
            context = context.unsqueeze(0)
        if context.shape[0] == 1 and batch_size > 1:
            context = context.repeat(batch_size, 1, 1)
        if context.shape[0] != batch_size:
            raise ValueError(f"Prompt context batch mismatch: {tuple(context.shape)} vs batch_size={batch_size}")
        return context

    @staticmethod
    def _first_tensor(value: Any) -> torch.Tensor:
        """"获取第一个 Tensor"""
        # Kept as a class method because Wan VAE may return Tensor or list[Tensor].
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"Expected one tensor from Wan VAE, got {len(value)}")
            value = value[0]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected Tensor, got {type(value)}.")
        return value

    def pad_and_encode(self, video: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """Pad and encode with the frozen VAE encoder."""
        # VAE encoder is frozen; no gradients are required here.
        with torch.no_grad():
            padded, pad_h, pad_w = self.pipe.pad_spatial(video)
            padded, pad_t = self.pipe.pad_temporal(padded)
            latent = self._first_tensor(self.pipe.encode_video(padded, **self.tiler_kwargs))
            latent = latent.to(device=self.device, dtype=self.torch_dtype)
        return latent, (pad_h, pad_w, pad_t)

    def decode_with_grad(self, latent: torch.Tensor, pad_info: Tuple[int, int, int]) -> torch.Tensor:
        """Decode with frozen VAE parameters while preserving dL/d(latent)."""
        # VAE params are frozen, but decoder graph must stay for image-domain losses.
        pad_h, pad_w, pad_t = pad_info
        video = self._first_tensor(self.pipe.decode_video(latent, **self.tiler_kwargs))
        video = self.pipe.unpad_temporal(video, pad_t)
        video = self.pipe.unpad_spatial(video, pad_h, pad_w)
        return video.contiguous()

    def forward_one_step(self, z_l: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable one-step DiT forward used by training."""
        timestep = torch.full((1,), self.args.fixed_timestep, dtype=self.torch_dtype, device=z_l.device)

        z_in = z_l
        if self.args.noise_step > 0:
            noise = torch.randn_like(z_l)
            noise_timestep = torch.full((1,), self.args.noise_step, dtype=self.torch_dtype, device=self.device)
            z_in = self.pipe.scheduler.add_noise(z_l, noise, timestep=noise_timestep)

        noise_pred = self.pipe.denoising_model()(
            z_in,
            timestep=timestep,
            context=context,
            **self.pipe.prepare_extra_input(z_in),
            use_gradient_checkpointing=self.args.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.args.use_gradient_checkpointing_offload,
        )
        z_out = z_in - noise_pred
        return z_out, noise_pred

    @staticmethod
    def _flatten_video(video: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = video.shape
        return (video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width).contiguous())

    def gaussian_high_frequency(self, video_01: torch.Tensor) -> torch.Tensor:
        """Compute gaussian high frequency"""
        batch, channels, frames, height, width = video_01.shape
        flat_frames = self._flatten_video(video_01)
        kernel = self.gaussian_kernel_5x5.to(device=flat_frames.device, dtype=flat_frames.dtype).repeat(channels, 1, 1, 1)

        blurred = F.conv2d(
            F.pad(flat_frames, (2, 2, 2, 2), mode="reflect"),
            kernel,
            groups=channels,
        )
        high = flat_frames - blurred
        return (high.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous())

    def compute_dists_loss(self, pred_video_01: torch.Tensor, target_video_01: torch.Tensor) -> torch.Tensor:
        if self.dists_loss is None:
            raise RuntimeError("DISTS is not initialized.")
        
        pred_frames = self._flatten_video(pred_video_01)
        target_frames = self._flatten_video(target_video_01)

        self.dists_loss.eval().float()
        with torch.autocast(device_type=pred_video_01.device.type, enabled=False):
            loss = self.dists_loss(pred_frames.float(), target_frames.float(),).mean()

        return loss.float()

    def compute_nqa_score(self, pred_video_01: torch.Tensor) -> torch.Tensor:
        if self.musiq is None:
            raise RuntimeError("MUSIQ is not initialized.")
        
        frames = self._flatten_video(pred_video_01)
        self.musiq.eval().float()

        with torch.autocast(device_type=pred_video_01.device.type, enabled=False):
            nqa_score = (self.musiq(frames.float()).mean() / 100.0)

        if not nqa_score.requires_grad:
            raise RuntimeError("MUSIQ score lost its input gradient.")
        return nqa_score.float()

    def sample_gan_frame_pairs(self, pred_video_m11: torch.Tensor, target_video_m11: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从视频中抽取相同位置的预测帧和 GT 帧。
        输入保持 VAE 图像域的 [-1, 1] 范围，避免 clamp 截断 GAN 对生成结果的梯度。
        """
        if pred_video_m11.shape != target_video_m11.shape:
            raise ValueError(
                "GAN input shape mismatch: "
                f"pred={tuple(pred_video_m11.shape)}, "
                f"target={tuple(target_video_m11.shape)}"
            )
        
        frame_count = pred_video_m11.shape[2]
        selected_count = min(self.args.gan_frames_per_video, frame_count)
        indices = torch.randperm(frame_count, device=pred_video_m11.device)[:selected_count]

        pred_frames = self._flatten_video(pred_video_m11.index_select(2, indices))
        target_frames = self._flatten_video(target_video_m11.index_select(2, indices))

        return pred_frames, target_frames

    def compute_gan_losses(self, pred_video_01: torch.Tensor, target_video_01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute hinge GAN losses with isolated generator/discriminator gradients.

        为了兼容 DeepSpeed，只使用了一个优化器。判别器在生成器前向传播过程中被冻结，而伪输入在其判别器前向传播过程中被分离，因此每个参数组只接收其自身的损失。
        """
        pred_frames, target_frames = self.sample_gan_frame_pairs(pred_video_01, target_video_01)

        discriminator_parameter = next(self.discriminator.parameters())
        discriminator_device = discriminator_parameter.device
        discriminator_dtype = discriminator_parameter.dtype

        pred_frames = pred_frames.to(device=discriminator_device, dtype=discriminator_dtype)
        target_frames = target_frames.to(device=discriminator_device, dtype=discriminator_dtype)

        self.discriminator.requires_grad_(False)
        try:
            fake_logits_for_generator = self.discriminator(pred_frames)
        finally:
            self.discriminator.requires_grad_(True)
        generator_loss = -fake_logits_for_generator.float().mean()

        real_logits = self.discriminator(target_frames.detach())
        fake_logits = self.discriminator(pred_frames.detach())
        discriminator_loss = 0.5 * (F.relu(1.0 - real_logits.float()).mean() + F.relu(1.0 + fake_logits.float()).mean())

        return generator_loss, discriminator_loss

    def _check_musiq_input_gradient(self) -> None:
        """
        检查 MUSIQ 模型的输出是否能够对输入图像反向传播梯度。 

        该检查只执行一次，主要用于确认 MUSIQ 可以作为可微分的质量损失。 
        如果 MUSIQ 输出与输入之间的计算图断开，或者输入梯度无效， 
        则提前抛出异常，避免训练过程中质量损失无法更新生成模型。
        """
        if self._musiq_gradient_checked:
            return
        if self.musiq is None:
            raise RuntimeError("MUSIQ is not initialized.")

        # 创建一张取值范围为 [0, 1) 的随机 RGB 图像。
        # shape: [B, C, H, W] = [1, 3, probe_height, probe_width]
        probe_height = min(self.args.height, 224)
        probe_width = min(self.args.width, 224)
        probe = torch.rand(1, 3, probe_height, probe_width, device=self.device, dtype=torch.float32, requires_grad=True)
        self.musiq.eval().float()

        with torch.autocast(device_type=self.device.type, enabled=False):
            score = self.musiq(probe).mean() / 100.0
        if not score.requires_grad:     # 如果输出不需要梯度，说明 MUSIQ 前向过程中计算图已经断开。此时 MUSIQ 分数不能作为损失反向更新输入图像或生成模型。
            raise RuntimeError("MUSIQ output does not require input gradients.")
        
        gradient = torch.autograd.grad(score, probe)[0]
        if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum().item() == 0.0:
            raise RuntimeError("MUSIQ input-gradient check failed.")
        self._musiq_gradient_checked = True
        py_logger.info("MUSIQ input-gradient check passed.")

    @staticmethod
    def _resize_flow_to_video(flow: torch.Tensor, height: int, width: int, expected_pairs: int, name: str) -> torch.Tensor:
        """
        将光流张量统一转换为 [B, 2, P, H, W] 格式， 并调整到目标视频的空间分辨率。 
        在调整光流图分辨率时，还会按宽高缩放比例同步调整 水平和垂直方向的光流位移值。
        Returns:
            torch.Tensor: [B, 2, expected_pairs, height, width]
        """
        if flow.dim() != 5:
            raise ValueError(f"Expected {name} as [B,2,T,H,W] or [B,T,2,H,W], got {tuple(flow.shape)}.")
        if flow.shape[1] == 2:
            normalized = flow
        elif flow.shape[2] == 2:
            normalized = flow.permute(0, 2, 1, 3, 4).contiguous()
        else:
            raise ValueError(f"Cannot find the two-channel flow dimension in {tuple(flow.shape)}.")

        if normalized.shape[2] < expected_pairs:
            raise ValueError(f"{name} has {normalized.shape[2]} pairs, expected {expected_pairs}.")
        normalized = normalized[:, :, :expected_pairs].contiguous()

        batch, channels, pairs, old_height, old_width = normalized.shape
        # [B, 2, P, H_old, W_old] -> [B*P, 2, C, H_old, W_old]
        flat = normalized.permute(0, 2, 1, 3, 4).reshape(batch * pairs, channels, old_height, old_width)
        flat = F.interpolate(flat, size=(height, width), mode="bilinear", align_corners=True)
        flat[:, 0] *= float(width) / max(float(old_width), 1.0)     # 缩放 x/y 轴的位移
        flat[:, 1] *= float(height) / max(float(old_height), 1.0)

        return flat.reshape(batch, pairs, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()


    @staticmethod
    def _in_bounds_mask(flow: torch.Tensor) -> torch.Tensor:
        """
        根据光流计算有效边界掩码。 

        对于每个像素位置 (x, y)，根据光流位移计算目标采样位置：sample_x = x + flow_x; sample_y = y + flow_y 
        如果目标位置仍位于图像范围内，则该位置的掩码值为 1； 如果目标位置越过图像边界，则掩码值为 0。

        Args: 
        flow: 光流张量，形状为 [B, 2, H, W]。 
        flow[:, 0] 表示水平方向位移 dx， flow[:, 1] 表示垂直方向位移 dy。 
        
        Returns: torch.Tensor: 有效边界掩码，形状为 [B, 1, H, W]， dtype 与输入 flow 相同。 
        """
        batch, _, height, width = flow.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=flow.device, dtype=flow.dtype),
            torch.arange(width, device=flow.device, dtype=flow.dtype),
            indexing="ij",
        )
        sample_x = grid_x.unsqueeze(0) + flow[:, 0]
        sample_y = grid_y.unsqueeze(0) + flow[:, 1]
        valid = (
            (sample_x >= 0.0)
            & (sample_x <= width - 1)
            & (sample_y >= 0.0)
            & (sample_y <= height - 1)
        )
        return valid.unsqueeze(1).to(flow.dtype)

    @staticmethod
    def _video_sample_indices(sample_type: Any, batch_size: int, device: torch.device) -> torch.Tensor:
        """根据 sample_type 判断 batch 中哪些样本属于视频，并返回这些视频样本在 batch 维度上的索引。"""
        if sample_type is None:                 # 没有提供样本类型信息，默认 batch 中 所有样本都是视频
            indices = list(range(batch_size))
        elif isinstance(sample_type, str):      # 整个 batch 共用一个样本类型。当值为 "video" 时，返回索引 [0]；否则返回空索引
            indices = (list(range(batch_size)) if sample_type == "video" else [])
        else:                                   # list、tuple 中每个元素对应 batch 中一个样本的类型，值等于 "video" 的位置会被保留
            indices = [index for index, value in enumerate(sample_type) if value == "video"]

        return torch.tensor(indices, device=device, dtype=torch.long)

    def compute_warp_loss(self, pred_video_01: torch.Tensor, sample_type: Any,) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算预测视频的高频光流对齐损失。
        
        首先使用双向 RAFT 为相邻视频帧估计前向和后向光流，然后利用指定方向的光流，将当前帧的高频分量 warp 到下一帧，并与下一帧的高频分量进行比较。
        只有同时满足以下条件的像素才参与损失计算：1. 前向、后向光流通过一致性检查； 2. 根据光流计算出的采样坐标位于图像边界内。

        Args: 
            pred_video_01: 预测视频张量，预期形状为： [B, C, T, H, W] 通常像素取值范围为 [0, 1]。
            sample_type: 当前 batch 中每个样本的类型信息，用于筛选其中的视频样本。例如： ["video", "image", "video"]

        Returns: Tuple[torch.Tensor, torch.Tensor]: 
            warp_loss: 所有相邻帧对上的平均高频光流对齐损失，标量 float32 张量。
            mask_ratio: 所有相邻帧对上的平均有效掩码比例，标量 float32 张量。
        """
        if self.raft is None:
            raise RuntimeError("RAFT is not initialized.")

        video_indices = self._video_sample_indices(sample_type, pred_video_01.shape[0], pred_video_01.device)
        if video_indices.numel() == 0 or pred_video_01.shape[2] < 2:    # 当前 batch 中没有视频样本/视频帧数小于 2，没有相邻帧对 无法计算
            zero = pred_video_01.new_tensor(0.0)
            return zero, zero

        video = pred_video_01.index_select(0, video_indices)    # [B_video, C, T, H, W]
        high_frequency = self.gaussian_high_frequency(video)
        _, channels, frames, height, width = video.shape

        with torch.no_grad():
            self.raft.eval().float()
            with torch.autocast(device_type=video.device.type, enabled=False):
                raft_output = self.raft(video.detach().float())

            if isinstance(raft_output, dict):
                flows_forward = None
                flows_backward = None
                for key in ("flows_forward", "forward", "flow_forward"):
                    if raft_output.get(key) is not None:
                        flows_forward = raft_output[key]
                        break
                for key in ("flows_backward", "backward", "flow_backward"):
                    if raft_output.get(key) is not None:
                        flows_backward = raft_output[key]
                        break
            elif isinstance(raft_output, (list, tuple)) and len(raft_output) == 2:
                flows_forward, flows_backward = raft_output
            else:
                raise RuntimeError(
                    "RAFT_bi must return a dict or a two-element tuple/list."
                )
            if flows_forward is None or flows_backward is None:
                raise RuntimeError("RAFT_bi did not return both flow directions.")

            # [B_video, 2, T - 1, H, W]
            flows_forward = self._resize_flow_to_video(
                flows_forward.detach().float(),
                height,
                width,
                expected_pairs=frames - 1,
                name="flows_forward",
            )
            flows_backward = self._resize_flow_to_video(
                flows_backward.detach().float(),
                height,
                width,
                expected_pairs=frames - 1,
                name="flows_backward",
            )

        # 当前帧 -> 下一帧。opposite_flows：用于进行前向-后向光流一致性检查。
        if self.args.flow_to_next_out == "backward":
            flows_to_next = flows_backward
            opposite_flows = flows_forward
        elif self.args.flow_to_next_out == "forward":
            flows_to_next = flows_forward
            opposite_flows = flows_backward
        else:
            raise ValueError(
                f"Unsupported FLOW_TO_NEXT_OUTPUT={self.args.flow_to_next_out!r}."
            )

        losses: List[torch.Tensor] = []
        mask_ratios: List[torch.Tensor] = []    # 保存每一对相邻帧中有效像素所占比例。
        for frame_index in range(frames - 1):
            flow_to_next = flows_to_next[:, :, frame_index]
            opposite_flow = opposite_flows[:, :, frame_index]

            consistency_mask = fbConsistencyCheck(flow_to_next, opposite_flow).to(device=video.device, dtype=high_frequency.dtype)
            in_bounds_mask = self._in_bounds_mask(flow_to_next).to(dtype=high_frequency.dtype)
            mask = consistency_mask * in_bounds_mask        # [B_video, 1, H, W]

            warped_current = flow_warp(high_frequency[:, :, frame_index], flow_to_next.permute(0, 2, 3, 1)) # [B_video, H, W, 2]
            difference = (warped_current - high_frequency[:, :, frame_index + 1]).abs()     # [B_video, C, H, W]
            denominator = mask.sum() * channels + 1e-6
            losses.append((difference * mask).sum() / denominator)
            mask_ratios.append(mask.mean())

        return (torch.stack(losses).mean().float(), torch.stack(mask_ratios).mean().float())

    def _log_training( self, loss: torch.Tensor, logs: Dict[str, torch.Tensor], batch_size: int) -> None:
        global_step = int(getattr(self.trainer, "global_step", 0) or 0)
        display_step = min(global_step + 1, self.args.max_steps)
        progress = 100.0 * display_step / max(float(self.args.max_steps), 1.0)

        self.log("Progress", progress, prog_bar=True, logger=False, on_step=True, on_epoch=False, sync_dist=False, rank_zero_only=True, batch_size=batch_size)
        
        self.log("train/loss", loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True, batch_size=batch_size)
        
        self.log_dict(logs, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=True, batch_size=batch_size)
        
        self.log(
            "train/lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
            rank_zero_only=True,
            batch_size=batch_size,
        )

        self.log(
            "train/lr_discriminator",
            self.trainer.optimizers[0].param_groups[1]["lr"],
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
            rank_zero_only=True,
            batch_size=batch_size,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        if self.dists_loss is None or self.musiq is None or self.raft is None:
            raise RuntimeError("TE losses were not initialized.")
        
        self.pipe.device = self.device

        # 每个训练 step 选择整个图像 batch 或整个视频 batch。
        is_image_batch = self.sample_image_batch()
        if is_image_batch:
            sample_type = "image"
            hq_input = batch["hq_image"]
            lq_input = batch["lq_image"]
        else:
            sample_type = "video"
            hq_input = batch["hq_video"]
            lq_input = batch["lq_video"]

        hq_input = hq_input.to(device=self.device, dtype=self.torch_dtype)
        lq_input = lq_input.to(device=self.device, dtype=self.torch_dtype)

        z_l, pad_info = self.pad_and_encode(lq_input)
        context = self.get_context(hq_input.shape[0])
        z_out, _ = self.forward_one_step(z_l, context)
        pred_output = self.decode_with_grad(z_out, pad_info)

        if pred_output.shape != hq_input.shape:
            raise ValueError(f"Output shape mismatch: pred={tuple(pred_output.shape)}, gt={tuple(hq_input.shape)}, sample_type={sample_type!r}.")

        if not torch.isfinite(pred_output).all():
            raise FloatingPointError("pred_output contains NaN or Inf before normalization.")

        if not torch.isfinite(hq_input).all():
            raise FloatingPointError("hq_input contains NaN or Inf.")

        pred_01 = ((pred_output.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        target_01 = ((hq_input.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        if not torch.isfinite(pred_01).all():
            raise FloatingPointError("pred_video contains NaN or Inf.")

        loss_l1 = F.l1_loss(pred_01, target_01, reduction="mean")
        loss_dists = self.compute_dists_loss(pred_01, target_01)
        loss_warp, flow_mask_ratio = self.compute_warp_loss(pred_01, sample_type)
        nqa_score = self.compute_nqa_score(pred_01)

        global_step = int(getattr(self.trainer, "global_step", 0) or 0)
        gan_scale = 0.0

        if global_step >= self.args.gan_start_step:
            if self.args.gan_warmup_steps > 0:
                gan_scale = min(float(global_step - self.args.gan_start_step + 1) / float(self.args.gan_warmup_steps), 1.0)
            else:
                gan_scale = 1.0
            loss_gan_g, loss_gan_d = self.compute_gan_losses(pred_output,hq_input)
        else:
            loss_gan_g = pred_01.new_zeros((), dtype=torch.float32)
            loss_gan_d = pred_01.new_zeros((), dtype=torch.float32)

        weighted_dists = self.args.lambda_dists * loss_dists
        weighted_warp = self.args.lambda_warp * loss_warp
        weighted_nqa = self.args.lambda_nqa * nqa_score
        weighted_gan = self.args.lambda_gan * gan_scale * loss_gan_g
        loss_generator = loss_l1 + weighted_dists + weighted_warp - weighted_nqa + weighted_gan
        loss = loss_generator + loss_gan_d

        self._log_training(
            loss,
            {
                "train/loss_l1": loss_l1.detach(),
                "train/loss_dists": loss_dists.detach(),
                "train/loss_warp": loss_warp.detach(),
                "train/nqa_score": nqa_score.detach(),
                "train/loss_gan_g": loss_gan_g.detach(),
                "train/loss_gan_d": loss_gan_d.detach(),
                "train/weighted_dists": weighted_dists.detach(),
                "train/weighted_warp": weighted_warp.detach(),
                "train/weighted_nqa": weighted_nqa.detach(),
                "train/weighted_gan": weighted_gan.detach(),
                "train/loss_generator": loss_generator.detach(),
                "train/gan_scale": pred_01.new_tensor(gan_scale, dtype=torch.float32,),
                "train/flow_mask_ratio": flow_mask_ratio.detach(),
                "train/is_image_batch": pred_01.new_tensor(float(is_image_batch)),
            },
            hq_input.shape[0],
        )
        return loss
    
    def _log_memory(self, tag: str, message: str) -> None:
        if not torch.cuda.is_available():
            return
        statistics = get_memory_statistics(device=self.device)
        if not statistics or not self.trainer.is_global_zero:
            return
        
        py_logger.info("%s: %s", message, json.dumps(statistics, indent=4))
        if self.logger is not None:
            metrics = {
                f"memory/{tag}/{key}": float(value)
                for key, value in statistics.items()
                if isinstance(value, (int, float))
            }
            self.logger.log_metrics(
                metrics,
                step=int(getattr(self.trainer, "global_step", 0) or 0),
            )

    def on_fit_start(self) -> None:
        if not self._init_checkpoint_loaded and not self.args.resume_from_checkpoint:
            raise RuntimeError("New Stage2-GAN training must use --init_from_checkpoint with a TC checkpoint or standard Stage2-TE checkpoint.")
        self._check_musiq_input_gradient()
        self._log_memory("Before_train", "Memory before training start")
        reset_peak_memory_stats(self.device)

    def on_train_start(self) -> None:
        self.pipe.vae.eval()
        self.pipe.denoising_model().train()
        for module in (self.dists_loss, self.musiq, self.raft):
            if module is not None:
                module.eval()
        self.discriminator.train()

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        freq = getattr(self.args, "statistic_frequency", 0)
        step = int(getattr(self.trainer, "global_step", 0) or 0)
        if freq > 0 and step > 0 and step % freq == 0 and step != self._last_memory_log_step:
            self._last_memory_log_step = step
            self._log_memory("After_iter", f"Memory after optimizer step {step} (frequency: {freq})")

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """在每次优化器更新参数之前，分别统计当前 DiT 和判别器的的梯度大小"""
        def gradient_statistics(parameters) -> Tuple[float, float]:
            total_norm_sq = 0.0         # 总梯度范数平方
            max_parameter_norm = 0.0    # 最大单参数梯度范数

            for parameter in parameters:
                if not parameter.requires_grad:
                    continue
                if parameter.grad is None:
                    continue

                parameter_norm = (parameter.grad.detach().float().norm(2).item())   # 单个参数张量的梯度 norm
                total_norm_sq += parameter_norm**2
                max_parameter_norm = max(max_parameter_norm, parameter_norm)

            return total_norm_sq**0.5, max_parameter_norm

        dit_grad_norm, dit_grad_norm_max = gradient_statistics(self.pipe.dit.parameters())
        discriminator_grad_norm, discriminator_grad_norm_max = gradient_statistics(self.discriminator.parameters())

        self.log_dict(
            {
                "train/grad_norm": dit_grad_norm,
                "train/grad_norm_max_param": dit_grad_norm_max,
                "train/grad_norm_discriminator": discriminator_grad_norm,
                "train/grad_norm_discriminator_max_param": discriminator_grad_norm_max,
            },
            logger=True,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=False,    # 记录单卡
            rank_zero_only=True,
        )

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        self.pipe.device = self.device
        self.pipe.denoising_model().eval()

        lq_path = batch["lq_path"][0] if isinstance(batch["lq_path"], (list, tuple)) else batch["lq_path"]
        hq_path = batch["hq_path"][0] if isinstance(batch["hq_path"], (list, tuple)) else batch["hq_path"]

        name = batch.get("name", [f"sample_{batch_idx:04d}"])
        name = name[0] if isinstance(name, (list, tuple)) else name

        lq_video, fps = prepare_input_tensor(
            lq_path, dtype=self.torch_dtype, device=self.device,
            spatial_scale=self.args.spatial_scale, temporal_scale=self.args.temporal_scale,
        )
        hq_video, _ = read_video_as_tensor(hq_path, dtype=self.torch_dtype, device=self.device)

        context = self.get_context(lq_video.shape[0])

        pred_video = self.pipe.test(
            lq_video=lq_video, prompt_emb_posi={"context": context},
            cfg_scale=1.0, num_inference_steps=1,
            fixed_timestep=self.args.fixed_timestep, noise_step=self.args.noise_step, color_fix=False, **self.tiler_kwargs,
        ).clamp(-1, 1)

        pred_fchw = bcthw_to_fchw_01(pred_video)
        target_fchw = bcthw_to_fchw_01(hq_video)

        if not self.metric_models:
            raise RuntimeError("Validation metrics are not initialized. They should be initialized in on_validation_start().")
        
        results = evaluate_video_metrics(
            pred_video=pred_fchw,
            ref_video=target_fchw,
            models=self.metric_models,
            device=self.device,
            name=name,
        )
        metric_logs = {
            f"val/{key}": float(value)
            for key, value in results.items()
            if isinstance(value, (int, float))
        }
        if metric_logs:
            self.log_dict(metric_logs, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)

        step = int(getattr(self.trainer, "global_step", 0) or 0)
        rank = int(self.global_rank)

        save_dir = Path(self.trainer.default_root_dir) / "validation" / f"step_{step:06d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_video(pred_video, str(save_dir / f"{name}_rank{rank}.mp4"),fps=self.args.fps)

        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe.denoising_model().train()
        return metric_logs

    def on_validation_start(self) -> None:
        if not self.metric_models:
            self._init_metrics()
        else:
            for metric in self.metric_models.values():
                metric.to(self.device)
                metric.eval()
        if not self.trainer.sanity_checking:
            self._log_memory("Before_val", "Memory before validation start")

    def on_validation_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        self._log_memory("After_val", "Memory after validation end")

        for metric in self.metric_models.values():
            metric.to("cpu")
            metric.eval()

        free_memory()
        self._log_memory("After_val_free", "Memory after moving validation metrics to CPU")
        reset_peak_memory_stats(self.device)

    def on_validation_epoch_end(self) -> None:
        # 只在主进程输出一次验证集平均指标
        if self.trainer.sanity_checking or not self.trainer.is_global_zero:
            return

        metrics: Dict[str, float] = {}
        for key, value in self.trainer.callback_metrics.items():
            if not isinstance(key, str) or not key.startswith("val/"):
                continue
            if torch.is_tensor(value):
                metrics[key] = float(value.detach().cpu())
            elif isinstance(value, (int, float)):
                metrics[key] = float(value)

        if metrics:
            step = int(getattr(self.trainer, "global_step", 0) or 0)
            metric_text = ", ".join(f"{key}: {value:.4f}" for key, value in sorted(metrics.items()))
            py_logger.info("[Validation Summary][step=%d] %s", step, metric_text)

    def configure_optimizers(self):
        dit_parameters = [
            parameter
            for parameter in self.pipe.dit.parameters()
            if parameter.requires_grad
        ]
        parameter_groups = [
            {
                "params": dit_parameters,
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
                "betas": (self.args.adam_beta1, self.args.adam_beta2,),
                "eps": self.args.adam_epsilon,
            },
            {
                "params": self.discriminator.parameters(),
                "lr": self.args.discriminator_learning_rate,
                "weight_decay": self.args.discriminator_weight_decay,
                "betas": (self.args.discriminator_beta1, self.args.discriminator_beta2),
                "eps": self.args.adam_epsilon,
            },
        ]

        optimizer = torch.optim.AdamW(parameter_groups)

        if self.args.lr_scheduler == "none":
            return optimizer
        
        if self.args.lr_scheduler == "constant_with_warmup":
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps)

        elif self.args.lr_scheduler == "decay_with_warmup":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.args.max_steps)

        elif self.args.lr_scheduler == "cosine_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.args.max_steps)
        
        else:
            raise ValueError(f"Unsupported lr_scheduler: {self.args.lr_scheduler}")

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        source_stage = checkpoint_stage(checkpoint)
        if source_stage != STAGE_NAME:
            raise ValueError(
                f"Stage2-GAN --resume_from_checkpoint requires a {STAGE_NAME!r} checkpoint, "
                f"but checkpoint stage is {source_stage!r}. "
                "Use --init_from_checkpoint when starting GAN training "
                "from Stage1-TC or standard Stage2-TE."
            )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Keep trainable DiT/GAN weights while preserving trainer states."""
        state_dict = checkpoint.get("state_dict", {})
        checkpoint["state_dict"] = {
            key: value
            for key, value in state_dict.items()
            if key.startswith("pipe.dit.") or key.startswith("discriminator.")
        }
        checkpoint["wan_stvsr_stage"] = STAGE_NAME
        checkpoint["wan_stvsr_args"] = vars(self.args)


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, Optional[DataLoader]]:
    train_dataset = SpatioTemporalRealSRImageVideoDataset(
        data_root=args.video_data_root,
        image_data_root=args.image_data_root,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        spatial_scale=args.spatial_scale,
        temporal_scale=args.temporal_scale,
        degradation_config=args.degradation_config,
        inter_frame_margin=args.inter_frame_margin,
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )

    val_loader: Optional[DataLoader] = None
    if args.val_lq_root and args.val_hq_root:
        val_dataset = PairedValidationVideoDataset(
            lq_root=args.val_lq_root,
            hq_root=args.val_hq_root,
            max_samples=args.max_val_samples,
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=1,
            num_workers=args.val_dataloader_num_workers,
            pin_memory=True,
            persistent_workers=args.val_dataloader_num_workers > 0,
        )
    return train_loader, val_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WanSTVSR Stage-2 TE pixel-domain training.")

    # Paths
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--video_data_root", type=str, required=True)
    parser.add_argument("--image_data_root", type=str, required=True)
    parser.add_argument("--val_lq_root", type=str, default=None)
    parser.add_argument("--val_hq_root", type=str, default=None)
    parser.add_argument("--degradation_config", type=str, required=True)
    parser.add_argument("--dit_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--text_encoder_path", type=str, default=None)
    parser.add_argument("--empty_prompt_embedding_path", type=str, default=None)
    parser.add_argument("--init_from_checkpoint", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--raft_ckpt_path", type=str, default="./utils/RAFT/raft-things.pth")

    # Dataset
    parser.add_argument("--num_frames", type=int, default=9)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--spatial_scale", type=int, default=4)
    parser.add_argument("--temporal_scale", type=int, default=2)
    parser.add_argument("--image_ratio", type=float, default=0.5)
    parser.add_argument("--inter_frame_margin", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--val_dataloader_num_workers", type=int, default=1)
    parser.add_argument("--max_val_samples", type=int, default=10)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)

    # Training
    parser.add_argument("--gpu_num", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5.0e-6)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup", choices=["none", "constant_with_warmup", "decay_with_warmup", "cosine_with_warmup"])
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--training_strategy", type=str, default="auto", choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"], help="Training strategy")
    parser.add_argument("--gradient_clip_val", type=float, default=0.0)
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--use_gradient_checkpointing_offload", action="store_true", default=False)

    # WanSTVSR one-step settings
    parser.add_argument("--fixed_timestep", type=int, default=799)
    parser.add_argument("--noise_step", type=int, default=0)
    parser.add_argument("--tiled", action="store_true", default=False)
    parser.add_argument("--tile_size_height", type=int, default=34)
    parser.add_argument("--tile_size_width", type=int, default=34)
    parser.add_argument("--tile_stride_height", type=int, default=18)
    parser.add_argument("--tile_stride_width", type=int, default=16)

    # Loss weights/Validation/logging/checkpoints
    parser.add_argument("--lambda_dists", type=float, default=1.0)
    parser.add_argument("--lambda_warp", type=float, default=0.05)
    parser.add_argument("--lambda_nqa", type=float, default=0.05)
    parser.add_argument("--lambda_gan", type=float, default=0.01)
    parser.add_argument("--gan_start_step", type=int, default=500)
    parser.add_argument("--gan_warmup_steps", type=int, default=500)
    parser.add_argument("--gan_frames_per_video", type=int, default=1)
    parser.add_argument("--discriminator_learning_rate", type=float, default=5e-5)
    parser.add_argument("--discriminator_weight_decay", type=float, default=0.0)
    parser.add_argument("--discriminator_beta1", type=float, default=0.0)
    parser.add_argument("--discriminator_beta2", type=float, default=0.99)
    parser.add_argument("--flow_to_next_out", type=str, default="backward", choices=[ "backward", "forward"])

    parser.add_argument("--val_check_interval", type=int, default=500)
    parser.add_argument("--statistic_frequency", type=int, default=500)
    parser.add_argument("--checkpoint_every_n_train_steps", type=int, default=1000)
    parser.add_argument("--metrics", type=str, default="psnr,ssim,lpips,dists,clipiqa")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--video_quality", type=int, default=9)
    parser.add_argument("--tensorboard_name", type=str, default="WanSTVSR_Stage2_GAN")

    args = parser.parse_args()
    if args.init_from_checkpoint and args.resume_from_checkpoint:
        raise ValueError(
            "--init_from_checkpoint and --resume_from_checkpoint are mutually exclusive."
        )
    if not args.init_from_checkpoint and not args.resume_from_checkpoint:
        raise ValueError(
            "Start Stage2-GAN with --init_from_checkpoint <stage1_or_stage2.ckpt>,"
            "or resume an interrupted Stage2-GAN run with --resume_from_checkpoint <stage2_gan.ckpt>."
        )
    if not 0.0 <= args.image_ratio <= 1.0:
        raise ValueError("--image_ratio must be in [0, 1].")
    if args.gan_frames_per_video <= 0:
        raise ValueError("--gan_frames_per_video must be positive.")

    if args.use_gradient_checkpointing_offload:
        args.use_gradient_checkpointing = True

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_path) / "args.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, ensure_ascii=False)
    return args


def train(args: argparse.Namespace) -> None:
    pl.seed_everything(args.seed, workers=True)
    train_loader, val_loader = build_dataloaders(args)
    dataset_size = len(train_loader.dataset)
    raw_train_batches = len(train_loader)

    global_batch_size = args.gpu_num * args.batch_size_per_gpu
    effective_batch_size = global_batch_size * args.accumulate_grad_batches

    usable_samples = (dataset_size // effective_batch_size) * effective_batch_size
    limit_train_batches = usable_samples // global_batch_size

    if limit_train_batches <= 0:
        raise ValueError(
            "The dataset is smaller than one effective batch: "
            f"dataset_size={dataset_size}, effective_batch_size={effective_batch_size}."
        )

    py_logger.info(
        f"Dataset size: {dataset_size}, "
        f"raw_train_batches_before_ddp={raw_train_batches}, "
        f"gpu_num={args.gpu_num}, "
        f"batch_size_per_gpu={args.batch_size_per_gpu}, "
        f"global_batch_size={global_batch_size}, "
        f"accumulate_grad_batches={args.accumulate_grad_batches}, "
        f"effective_batch_size={effective_batch_size}, "
        f"usable_samples={usable_samples}, "
        f"limit_train_batches_per_rank={limit_train_batches}, "
        f"optimizer_steps_per_epoch={limit_train_batches // args.accumulate_grad_batches}."
    )

    model = LightningModelForStage2(args)
    logger = TensorBoardLogger(
        save_dir=str(Path(args.output_path) / "tensorboardlog"),
        name=args.tensorboard_name,
        default_hp_metric=False,
    )
    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
        save_top_k=-1,
        every_n_train_steps=args.checkpoint_every_n_train_steps,
    )

    val_check_interval = None
    if val_loader is not None and args.val_check_interval > 0:
        val_check_interval = (
            args.val_check_interval * args.accumulate_grad_batches
        )

    trainer = pl.Trainer(
        max_steps=args.max_steps,
        accelerator="gpu",
        devices=args.gpu_num,
        strategy=args.training_strategy,
        precision="bf16",
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=[checkpoint_callback],
        logger=[logger],
        log_every_n_steps=1,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=None if val_loader is not None else 1,  # 不按 epoch 进行验证
        num_sanity_val_steps=0,
        use_distributed_sampler=True,
        limit_train_batches=limit_train_batches,
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from_checkpoint)

    free_memory()
    if trainer.is_global_zero:
        memory_statistics = get_memory_statistics()
        py_logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")


if __name__ == "__main__":
    train(parse_args())
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WanSTVSR training script.

Design:
  - TC trains latent reconstruction and temporal residual consistency.
  - TE trains pixel fidelity, DISTS, high-frequency warp, and MUSIQ losses.
  - VAE/text encoder/RAFT/metric networks are frozen; only DiT is optimized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.append('/data2/wujialing/project/STVSR/WanSTVSR-0713')

import lightning as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

import pyiqa

from diffsynth import ModelManager
from diffsynth.pipelines.wan_stvsr import WanSTVSRPipeline
from dataset.spatio_temporal_real_sr_image_video_dataset import (
    SpatioTemporalRealSRImageVideoDataset,
)
from dataset.spatio_temporal_real_sr_dataset import SpatioTemporalRealSRDataset
from dataset.utils import scan_video_or_frame_dirs, read_video_as_tensor, prepare_input_tensor, bcthw_to_fchw_01, save_video

from utils.metric_utils import evaluate_video_metrics
from utils.memory_utils import free_memory, get_memory_statistics, reset_peak_memory_stats
from utils.optical_flow_utils import flow_warp, fbConsistencyCheck
from utils.RAFT.raft_bi import RAFT_bi

import logging

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stdout,)
py_logger = logging.getLogger(__name__)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class PairedValidationVideoDataset(Dataset):
    """将验证集的 LQ 和 HQ 配对，返回路径"""

    def __init__(self, lq_root, hq_root, max_samples: Optional[int] = None):
        super().__init__()
        self.lq_root = Path(lq_root)
        self.hq_root = Path(hq_root)

        lq_samples = scan_video_or_frame_dirs(self.lq_root)
        hq_samples = scan_video_or_frame_dirs(self.hq_root)
        hq_by_stem = {p.stem: p for p in hq_samples}    # 将 hq 样本按 stem 索引

        pairs: List[Tuple[Path, Path]] = []
        for lq_path in lq_samples:
            hq_path = hq_by_stem.get(lq_path.stem)
            if hq_path is not None:
                pairs.append((lq_path, hq_path))    # 建立验证样本对

        if max_samples is not None and max_samples > 0: # 限制样本数量
            pairs = pairs[:max_samples]
        if not pairs:
            raise FileNotFoundError(f"No paired validation samples found in {self.lq_root} and {self.hq_root}")
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        lq_path, hq_path = self.pairs[idx]
        return {"lq_path": str(lq_path), "hq_path": str(hq_path), "name": lq_path.stem}


class LightningModelForTrain(pl.LightningModule):
    strict_loading = False

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.save_hyperparameters(vars(args))   # 把训练参数保存进 checkpoint 的 hyper_parameters 里，方便恢复、记录实验配置
        self.args = args
        self.torch_dtype = torch.bfloat16
        self.tiler_kwargs = {
            "tiled": args.tiled,
            "tile_size": (args.tile_size_height, args.tile_size_width),
            "tile_stride": (args.tile_stride_height, args.tile_stride_width),
        }

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
        self.musiq: Optional[Any] = None
        self.metric_models: Dict[str, Any] = {}
        self.raft: Optional[Any] = None
        self._musiq_gradient_checked = False
        self._flow_direction_checked = False

        coordinates = torch.arange(5, dtype=torch.float32) - 2.0
        gaussian_1d = torch.exp(-(coordinates ** 2) / 2.0)
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        self.register_buffer("gaussian_kernel_5x5", gaussian_2d.view(1, 1, 5, 5), persistent=False)

        # 避免梯度累积时，同一个 global_step 被 on_train_batch_end 重复记录
        self._last_memory_log_step = -1
    def freeze_parameters(self) -> None:
        # Freeze the full pipeline first, then enable the trainable part.
        self.pipe.requires_grad_(False)
        self.pipe.eval()

        self.pipe.denoising_model().requires_grad_(True)
        self.pipe.denoising_model().train()

        # Keep trainable DiT params fp32 for optimizer stability.
        for param in self.pipe.dit.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

    def _load_initial_dit(self, checkpoint_path: str) -> None:
        """Load only DiT parameters; optimizer/step state is intentionally ignored."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            raise TypeError(f"Unsupported checkpoint state type: {type(state_dict)}")

        dit_state = {}
        for key, value in state_dict.items():
            if key.startswith("pipe.dit."):
                dit_state[key[len("pipe.dit."):]] = value
            elif key.startswith("dit."):
                dit_state[key[len("dit."):]] = value
        if not dit_state:
            raise ValueError(
                f"No DiT keys with prefix 'pipe.dit.' or 'dit.' found in {checkpoint_path}"
            )

        incompatible = self.pipe.dit.load_state_dict(dit_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"DiT checkpoint mismatch: {incompatible}")

        source_args = checkpoint.get("wan_stvsr_args", {}) if isinstance(checkpoint, dict) else {}
        source_stage = source_args.get("train_stage") if isinstance(source_args, dict) else None
        if self.args.train_stage == "te" and source_stage != "tc":
            raise ValueError(
                f"TE must initialize from a TC checkpoint, but source train_stage={source_stage!r}"
            )
        self._init_checkpoint_loaded = True
        py_logger.info("Loaded DiT-only initialization from %s", checkpoint_path)

    def setup(self, stage: Optional[str] = None) -> None:
        """"准备 prompt embedding，初始化训练 loss 和验证 metrics"""
        if self.context is None:
            self.context = self._load_or_encode_context()
        if stage in ("fit", None):
            self._init_losses()
        if self.args.train_stage == "te" and stage in ("fit", "validate", "test", None):
            self._init_metrics()

    def _load_or_encode_context(self) -> torch.Tensor:
        if self.args.empty_prompt_embedding_path:
            # Load empty prompt embedding，兼容多种格式
            data = torch.load(self.args.empty_prompt_embedding_path, map_location="cpu", weights_only=False)    # 文件本身就是 tensor
            emb = data.get("prompt_emb", data) if isinstance(data, dict) else data      # 文件是 dict，里面有 prompt_emb
            if isinstance(emb, dict):           # 文件是 dict，prompt_emb 里面又是 dict
                emb = emb["context"]
            return emb.detach().cpu()

        # 没有 embedding 文件，使用 text encoder 编码 prompt
        if self.pipe.text_encoder is None:
            raise ValueError("Provide --empty_prompt_embedding_path or --text_encoder_path")

        old_device = self.pipe.device
        self.pipe.device = self.device
        self.pipe.text_encoder.to(self.device, dtype=self.torch_dtype).eval()
        with torch.no_grad():
            emb = self.pipe.encode_prompt(self.prompt_text, positive=True)["context"].detach().cpu()
        self.pipe.text_encoder.to("cpu")
        self.pipe.device = old_device
        free_memory()
        return emb

    def _init_losses(self) -> None:
        if self.args.train_stage == "tc":
            if any(module is not None for module in (self.dists_loss, self.musiq, self.raft)):
                raise RuntimeError("TC must not initialize DISTS, MUSIQ, or RAFT")
            return

        if pyiqa is None:
            raise ImportError("pyiqa is required for TE DISTS and MUSIQ losses")
        self.dists_loss = pyiqa.create_metric(
            "dists", device=self.device, as_loss=True
        ).eval()
        self.musiq = pyiqa.create_metric("musiq", device=self.device).eval()
        for module in (self.dists_loss, self.musiq):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.raft = RAFT_bi(self.args.raft_ckpt_path, device=self.device)
        self.raft.requires_grad_(False)
        self.raft.eval()
        self._validate_flow_warp_direction()

    def _init_metrics(self) -> None:
        if pyiqa is None:
            raise ImportError("pyiqa is required for validation metrics")
        metric_names = [m.strip().lower() for m in self.args.metrics.split(",") if m.strip()]
        if self.args.train_stage == "tc":
            metric_names = [name for name in metric_names if name not in {"dists", "musiq"}]
        for name in metric_names:
            if name in self.metric_models:
                continue
            self.metric_models[name] = pyiqa.create_metric(name, device=self.device).eval()
            if hasattr(self.metric_models[name], "parameters"):
                for parameter in self.metric_models[name].parameters():
                    parameter.requires_grad_(False)

    def get_context(self, batch_size: int) -> torch.Tensor:
        """
        处理编码好的 context embedding 到 DiT forward 需要的形状 [B, L, C]"""
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

    def _first_tensor(self, value: Any) -> torch.Tensor:
        """"获取第一个 Tensor"""
        # Kept as a class method because Wan VAE may return Tensor or list[Tensor].
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"Expected one tensor from Wan VAE, got {len(value)}")
            return value[0]
        return value

    def _log_memory(self, tag: str, message: str) -> None:
        if not torch.cuda.is_available():
            return

        memory_statistics = get_memory_statistics(device=self.device)
        if not memory_statistics:
            return
        
        step = int(getattr(self.trainer, "global_step", 0) or 0)

        if self.trainer.is_global_zero:
            py_logger.info(f"{message}: {json.dumps(memory_statistics, indent=4)}")

            if self.logger is not None:
                metrics = {
                    f"memory/{tag}/{k}": float(v)
                    for k, v in memory_statistics.items()
                    if isinstance(v, (int, float))
                }
                self.logger.log_metrics(metrics, step=step)

    def on_fit_start(self) -> None:
        if self.args.train_stage == "tc":
            if any(module is not None for module in (self.dists_loss, self.musiq, self.raft)):
                raise RuntimeError("TC unexpectedly initialized a TE-only loss module")
        else:
            if not self._init_checkpoint_loaded and not self.args.resume_from_checkpoint:
                raise RuntimeError("New TE training must load Stage-1 DiT weights with --init_from_checkpoint")
            self._check_musiq_input_gradient()

        self._log_memory("Before_train", "Memory before training start")
        reset_peak_memory_stats(self.device)

    def on_train_start(self) -> None:
        self.pipe.vae.eval()
        self.pipe.denoising_model().train()
        for module in (self.dists_loss, self.musiq, self.raft):
            if module is not None:
                module.eval()

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        freq = getattr(self.args, "statistic_frequency", 0)
        step = int(getattr(self.trainer, "global_step", 0) or 0)

        # 普通文本进度日志，写到终端和 train.log
        # if self.trainer.is_global_zero and step > 0 and step % 10 == 0:
        #     lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        #     loss = self.trainer.callback_metrics.get("train/loss", None)
        #     if torch.is_tensor(loss):
        #         loss = float(loss.detach().cpu())
        #         py_logger.info(
        #             f"[Train] step={step}/{self.args.max_steps}, "
        #             f"loss={loss:.6f}, lr={lr:.3e}"
        #         )
        #     else:
        #         py_logger.info(
        #             f"[Train] step={step}/{self.args.max_steps}, lr={lr:.3e}"
        #         )

        # 显存统计日志
        if freq > 0 and step > 0 and step % freq == 0 and step != self._last_memory_log_step:
            self._last_memory_log_step = step
            self._log_memory("After_iter", f"Memory after optimizer step {step} (frequency: {freq})")

    def on_validation_start(self) -> None:
        if self.trainer.sanity_checking:
            return
        self._log_memory("Before_val", "Memory before validation start")

    def on_validation_end(self) -> None:
        if self.trainer.sanity_checking:
            return

        self._log_memory("After_val", "Memory after validation end")
        free_memory()
        self._log_memory("After_val_free", "Memory after validation free_memory")
        reset_peak_memory_stats(self.device)
    
    def on_before_optimizer_step(self, optimizer) -> None:
        """在每次优化器更新参数之前，统计当前 DiT 参数的梯度大小"""
        total_norm_sq = 0.0     # 总梯度范数平方
        max_param_norm = 0.0    # 最大单参数梯度范数

        for p in self.pipe.dit.parameters():
            if p.requires_grad and p.grad is not None:
                param_norm = p.grad.detach().float().norm(2).item()     # 单个参数张量的梯度 norm
                total_norm_sq += param_norm ** 2
                max_param_norm = max(max_param_norm, param_norm)

        total_norm = total_norm_sq ** 0.5

        self.log_dict(
            {
                "train/grad_norm": total_norm,
                "train/grad_norm_max_param": max_param_norm,
            },
            logger=True,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=False,    # 记录单卡
            rank_zero_only=True,
        )
        
    def pad_and_encode(self, video: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        # VAE encoder is frozen; no gradients are required here.
        with torch.no_grad():
            video_pad, pad_h, pad_w = self.pipe.pad_spatial(video)
            video_pad, pad_t = self.pipe.pad_temporal(video_pad)
            latent = self._first_tensor(self.pipe.encode_video(video_pad, **self.tiler_kwargs))
            latent = latent.to(device=self.device, dtype=self.torch_dtype)
        return latent, (pad_h, pad_w, pad_t)

    def _resize_flow_to_video(self, flow: torch.Tensor, height: int, width: int, expected_pairs: Optional[int] = None, name: str = "flow") -> torch.Tensor:
        """Normalize and resize optical flow to video resolution.
            Accepts:
            - [B, 2, T, H, W]
            - [B, T, 2, H, W]

            Returns:
            - [B, 2, T, height, width]
        """
        # 2：光流的 x/y 位移，通常第 0 通道是水平方向，第 1 通道是垂直方向
        if not isinstance(flow, torch.Tensor):
            raise TypeError(f"{name} must be a Tensor, got {type(flow)}")

        if flow.dim() != 5:
            raise ValueError(f"Expected {name} as [B,2,T,H,W] or [B,T,2,H,W], got {tuple(flow.shape)}")
        
        # Normalize to [B, 2, T, H, W]
        if flow.shape[1] == 2:
            pass
        elif flow.shape[2] == 2:
            flow = flow.permute(0, 2, 1, 3, 4).contiguous()
        else:
            raise ValueError(
                f"Cannot identify flow channel dimension for {name}, got {tuple(flow.shape)}. "
                "Expected channel dimension with size 2."
            )

        if expected_pairs is not None:      # 判断 T 是否大于光流对数
            if flow.shape[2] < expected_pairs:
                raise ValueError(
                    f"{name} temporal length is too short: got {flow.shape[2]}, "
                    f"expected at least {expected_pairs} for video T={expected_pairs + 1}."
                )
            if flow.shape[2] > expected_pairs:
                flow = flow[:, :, :expected_pairs].contiguous()

        b, c, t_flow, h0, w0 = flow.shape
        if c != 2:
            raise ValueError(f"Expected {name} channel=2, got {tuple(flow.shape)}")
        
        flow_4d = flow.permute(0, 2, 1, 3, 4).reshape(b * t_flow, c, h0, w0)     # [B*T, C, Hf, Wf]
        flow_4d = F.interpolate(flow_4d, size=(height, width), mode="bilinear", align_corners=True)

        # x/y displacement also needs to be scaled after resizing.
        flow_4d[:, 0] *= float(width) / max(float(w0), 1.0)
        flow_4d[:, 1] *= float(height) / max(float(h0), 1.0)

        return flow_4d.reshape(b, t_flow, c, height, width).permute(0, 2, 1, 3, 4).contiguous()  # [B,C,T,H,W]


    @torch.no_grad()
    def _validate_flow_warp_direction(self) -> None:
        """Verify backward-warp sign with a one-pixel horizontal translation."""
        source = torch.zeros(1, 1, 5, 7, device=self.device, dtype=torch.float32)
        target = torch.zeros_like(source)
        source[:, :, 2, 2] = 1.0
        target[:, :, 2, 3] = 1.0

        flow_target_to_source = torch.zeros(1, 5, 7, 2, device=self.device)
        flow_target_to_source[..., 0] = -1.0
        flow_source_to_target = torch.zeros_like(flow_target_to_source)
        flow_source_to_target[..., 0] = 1.0

        aligned = flow_warp(source, flow_target_to_source)
        wrong = flow_warp(source, flow_source_to_target)
        if not torch.allclose(aligned, target, atol=1e-6, rtol=0.0) or torch.allclose(
            wrong, target, atol=1e-6, rtol=0.0
        ):
            raise RuntimeError("flow_warp direction test failed")
        self._flow_direction_checked = True

    def _check_musiq_input_gradient(self) -> None:
        if self._musiq_gradient_checked:
            return
        if self.musiq is None:
            raise RuntimeError("MUSIQ is not initialized for TE")
        probe = torch.rand(
            1, 3, self.args.height, self.args.width,
            device=self.device, dtype=torch.float32, requires_grad=True,
        )
        self.musiq.eval()
        self.musiq.float()
        with torch.autocast(device_type=self.device.type, enabled=False):
            score = self.musiq(probe).mean() / 100.0
        if not score.requires_grad:
            raise RuntimeError("MUSIQ score does not require gradients from its input")
        gradient = torch.autograd.grad(score, probe, retain_graph=False, create_graph=False)[0]
        if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
            raise RuntimeError("MUSIQ failed the input-gradient check")
        self._musiq_gradient_checked = True
        py_logger.info("MUSIQ input-gradient check passed")

    @staticmethod
    def _assert_unit_range(video: torch.Tensor, name: str) -> None:
        minimum = float(video.detach().amin().cpu())
        maximum = float(video.detach().amax().cpu())
        if minimum < -1e-6 or maximum > 1.0 + 1e-6:
            raise ValueError(f"{name} must be in [0,1], got range [{minimum}, {maximum}]")

    @staticmethod
    def _flatten_video(video: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = video.shape
        return video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).contiguous()

    def gaussian_high_frequency(self, video_01: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = video_01.shape
        frames = self._flatten_video(video_01)
        kernel = self.gaussian_kernel_5x5.to(device=frames.device, dtype=frames.dtype)
        kernel = kernel.repeat(c, 1, 1, 1)
        blurred = F.conv2d(F.pad(frames, (2, 2, 2, 2), mode="reflect"), kernel, groups=c)
        high = frames - blurred
        return high.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

    def compute_dists_loss(
        self, pred_video_01: torch.Tensor, target_video_01: torch.Tensor
    ) -> torch.Tensor:
        if self.dists_loss is None:
            raise RuntimeError("DISTS is not initialized for TE")
        pred_frames = self._flatten_video(pred_video_01)
        target_frames = self._flatten_video(target_video_01)
        self.dists_loss.eval()
        self.dists_loss.float()
        with torch.autocast(device_type=pred_video_01.device.type, enabled=False):
            return self.dists_loss(pred_frames.float(), target_frames.float()).mean().float()

    def compute_nqa_score(self, pred_video_01: torch.Tensor) -> torch.Tensor:
        if self.musiq is None:
            raise RuntimeError("MUSIQ is not initialized for TE")
        frames = self._flatten_video(pred_video_01)
        self.musiq.eval()
        self.musiq.float()
        scores = []
        with torch.autocast(device_type=pred_video_01.device.type, enabled=False):
            for start in range(0, frames.shape[0], self.args.nqa_chunk_size):
                scores.append(self.musiq(frames[start:start + self.args.nqa_chunk_size].float()).reshape(-1))
        nqa_score = torch.cat(scores).mean() / 100.0
        if not nqa_score.requires_grad:
            raise RuntimeError("MUSIQ score lost its gradient to the predicted video")
        return nqa_score.float()

    @staticmethod
    def _in_bounds_mask(flow: torch.Tensor) -> torch.Tensor:
        b, _, h, w = flow.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=flow.device, dtype=flow.dtype),
            torch.arange(w, device=flow.device, dtype=flow.dtype),
            indexing="ij",
        )
        sample_x = grid_x.unsqueeze(0) + flow[:, 0]
        sample_y = grid_y.unsqueeze(0) + flow[:, 1]
        valid = (
            (sample_x >= 0.0) & (sample_x <= w - 1)
            & (sample_y >= 0.0) & (sample_y <= h - 1)
        )
        return valid.unsqueeze(1).to(flow.dtype)

    @staticmethod
    def _video_sample_indices(sample_type: Any, batch_size: int, device: torch.device) -> torch.Tensor:
        if sample_type is None:
            indices = list(range(batch_size))
        elif isinstance(sample_type, str):
            indices = [0] if sample_type == "video" else []
        else:
            indices = [index for index, value in enumerate(sample_type) if value == "video"]
        return torch.tensor(indices, device=device, dtype=torch.long)

    def compute_warp_loss(
        self, pred_video_01: torch.Tensor, sample_type: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.raft is None or not self._flow_direction_checked:
            raise RuntimeError("RAFT or flow direction check is not ready for TE")
        indices = self._video_sample_indices(
            sample_type, pred_video_01.shape[0], pred_video_01.device
        )
        if indices.numel() == 0 or pred_video_01.shape[2] < 2:
            zero = pred_video_01.new_tensor(0.0)
            return zero, zero

        video = pred_video_01.index_select(0, indices)
        high_frequency = self.gaussian_high_frequency(video)
        _, channels, frames, height, width = video.shape

        with torch.no_grad():
            self.raft.eval()
            self.raft.float()
            with torch.autocast(device_type=video.device.type, enabled=False):
                flows_forward, flows_backward = self.raft(video.detach().float())
            flows_forward = self._resize_flow_to_video(
                flows_forward.detach().float(), height, width,
                expected_pairs=frames - 1, name="flows_forward",
            )
            flows_backward = self._resize_flow_to_video(
                flows_backward.detach().float(), height, width,
                expected_pairs=frames - 1, name="flows_backward",
            )

        losses: List[torch.Tensor] = []
        mask_ratios: List[torch.Tensor] = []
        for index in range(frames - 1):
            # RAFT_bi calls RAFT(I_j, I_{j+1}) for forward and the reverse
            # pair for backward. flow_warp samples source I_j at target
            # I_{j+1} coordinates, so it requires the latter flow.
            flow_forward = flows_forward[:, :, index]
            flow_to_next = flows_backward[:, :, index]
            consistency = fbConsistencyCheck(flow_to_next, flow_forward)
            in_bounds = self._in_bounds_mask(flow_to_next)
            mask = (consistency * in_bounds).to(high_frequency.dtype)

            warped = flow_warp(
                high_frequency[:, :, index],
                flow_to_next.permute(0, 2, 3, 1),
            )
            difference = (warped - high_frequency[:, :, index + 1]).abs()
            denominator = mask.sum() * channels + 1e-6
            losses.append((difference * mask).sum() / denominator)
            mask_ratios.append(mask.mean())

        return torch.stack(losses).mean().float(), torch.stack(mask_ratios).mean().float()

    def _call_pipeline(self, lq_video: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        z_out, _ = self.pipe.call(
            lq_video=lq_video,
            context=context,
            fixed_timestep=self.args.fixed_timestep,
            noise_step=self.args.noise_step,
            **self.tiler_kwargs,
            use_gradient_checkpointing=self.args.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.args.use_gradient_checkpointing_offload,
        )
        return z_out

    def _log_training(
        self, loss: torch.Tensor, logs: Dict[str, torch.Tensor], batch_size: int
    ) -> None:
        completed_step = int(getattr(self.trainer, "global_step", 0) or 0)
        display_step = min(completed_step + 1, self.args.max_steps)
        progress_pct = 100.0 * display_step / max(float(self.args.max_steps), 1.0)
        self.log(
            "Progress", progress_pct, prog_bar=True, logger=False, on_step=True,
            on_epoch=False, sync_dist=False, rank_zero_only=True, batch_size=batch_size,
        )
        self.log(
            "train/loss", loss.detach(), prog_bar=True, logger=True, on_step=True,
            on_epoch=False, sync_dist=True, batch_size=batch_size,
        )
        self.log_dict(
            logs, prog_bar=False, logger=True, on_step=True, on_epoch=False,
            sync_dist=True, batch_size=batch_size,
        )
        self.log(
            "train/lr", self.trainer.optimizers[0].param_groups[0]["lr"],
            prog_bar=False, logger=True, on_step=True, on_epoch=False,
            sync_dist=False, rank_zero_only=True, batch_size=batch_size,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        if self.args.train_stage == "tc":
            return self.training_step_tc(batch, batch_idx)
        return self.training_step_te(batch, batch_idx)

    def training_step_tc(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        if any(module is not None for module in (self.dists_loss, self.musiq, self.raft)):
            raise RuntimeError("TC must not load DISTS, MUSIQ, or RAFT")
        self.pipe.device = self.device
        hq_video = batch["hq_video"].to(device=self.device, dtype=self.torch_dtype)
        lq_video = batch["lq_video"].to(device=self.device, dtype=self.torch_dtype)
        z_gt, _ = self.pad_and_encode(hq_video)
        z_out = self._call_pipeline(lq_video, self.get_context(hq_video.shape[0]))

        if z_out.shape != z_gt.shape:
            raise ValueError(f"Latent shape mismatch: z_out={z_out.shape}, z_gt={z_gt.shape}")
        if z_out.shape[2] <= 1:
            raise ValueError(f"TC latent temporal length must be > 1, got {z_out.shape[2]}")

        loss_mse = F.mse_loss(z_out.float(), z_gt.float())
        delta_out = z_out[:, :, 1:] - z_out[:, :, :-1]
        delta_gt = z_gt[:, :, 1:] - z_gt[:, :, :-1]
        loss_res = F.l1_loss(delta_out.float(), delta_gt.float(), reduction="mean")
        loss = loss_mse + self.args.lambda_res * loss_res
        logs = {
            "train/loss_mse": loss_mse.detach(),
            "train/loss_res": loss_res.detach(),
            "train/latent_length": z_out.new_tensor(float(z_out.shape[2])),
        }
        self._log_training(loss, logs, hq_video.shape[0])
        return loss

    def training_step_te(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        if self.dists_loss is None or self.musiq is None or self.raft is None:
            raise RuntimeError("TE requires initialized DISTS, MUSIQ, and RAFT")
        self.pipe.device = self.device
        hq_video = batch["hq_video"].to(device=self.device, dtype=self.torch_dtype)
        lq_video = batch["lq_video"].to(device=self.device, dtype=self.torch_dtype)
        z_out = self._call_pipeline(lq_video, self.get_context(hq_video.shape[0]))
        pred_video = self.pipe.call_decode_seg(
            z_out, output_shape=tuple(hq_video.shape[-3:]), **self.tiler_kwargs
        )
        if pred_video.shape != hq_video.shape:
            raise ValueError(f"Video shape mismatch: pred={pred_video.shape}, gt={hq_video.shape}")

        pred_01 = ((pred_video.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        target_01 = ((hq_video.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        self._assert_unit_range(pred_01, "pred_video_01")
        self._assert_unit_range(target_01, "target_video_01")

        loss_l1 = F.l1_loss(pred_01, target_01, reduction="mean")
        loss_dists = self.compute_dists_loss(pred_01, target_01)
        loss_warp, flow_mask_ratio = self.compute_warp_loss(
            pred_01, batch.get("sample_type")
        )
        nqa_score = self.compute_nqa_score(pred_01)

        weighted_dists = self.args.lambda_dists * loss_dists
        weighted_warp = self.args.lambda_warp * loss_warp
        weighted_nqa = self.args.lambda_nqa * nqa_score
        loss = loss_l1 + weighted_dists + weighted_warp - weighted_nqa
        logs = {
            "train/loss_l1": loss_l1.detach(),
            "train/loss_dists": loss_dists.detach(),
            "train/loss_warp": loss_warp.detach(),
            "train/nqa_score": nqa_score.detach(),
            "train/weighted_dists": weighted_dists.detach(),
            "train/weighted_warp": weighted_warp.detach(),
            "train/weighted_nqa": weighted_nqa.detach(),
            "train/flow_mask_ratio": flow_mask_ratio.detach(),
        }
        self._log_training(loss, logs, hq_video.shape[0])
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        self.pipe.device = self.device
        self.pipe.denoising_model().eval()

        lq_path = batch["lq_path"][0] if isinstance(batch["lq_path"], (list, tuple)) else batch["lq_path"]
        hq_path = batch["hq_path"][0] if isinstance(batch["hq_path"], (list, tuple)) else batch["hq_path"]

        name = batch.get("name", [f"sample_{batch_idx:04d}"])
        name = name[0] if isinstance(name, (list, tuple)) else name

        if isinstance(batch.get("lq_path", None), (list, tuple)):
            batch_size = len(batch["lq_path"])
        elif isinstance(batch.get("name", None), (list, tuple)):
            batch_size = len(batch["name"])
        else:
            batch_size = 1

        lq_video, fps = prepare_input_tensor(
            lq_path, dtype=self.torch_dtype, device=self.device,
            spatial_scale=self.args.spatial_scale, temporal_scale=self.args.temporal_scale,
        )
        hq_video, _ = read_video_as_tensor(hq_path, dtype=self.torch_dtype, device=self.device)

        context = self.get_context(batch_size=lq_video.shape[0])    # bs = 1

        pred_video = self.pipe.test(
            lq_video=lq_video, prompt_emb_posi={"context": context},
            cfg_scale=1.0, num_inference_steps=1,
            fixed_timestep=self.args.fixed_timestep, noise_step=self.args.noise_step, color_fix=True,
            tiled=self.args.tiled, tile_size=(self.args.tile_size_height, self.args.tile_size_width), tile_stride=(self.args.tile_stride_height, self.args.tile_stride_width),
        ).clamp(-1, 1)

        pred_fchw = bcthw_to_fchw_01(pred_video)
        target_fchw = bcthw_to_fchw_01(hq_video)

        metric_logs: Dict[str, float] = {}
        if self.metric_models:
            results = evaluate_video_metrics(
                pred_video=pred_fchw,
                ref_video=target_fchw,
                models=self.metric_models,
                device=self.device,
                name=name,
            )
            if isinstance(results, dict):
                metric_logs = {f"val/{k}": float(v) for k, v in results.items() if isinstance(v, (int, float))}
                self.log_dict(metric_logs, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)    # 上传到 Tensorboard / Lightning logger

        global_step = int(getattr(self.trainer, "global_step", 0) or 0)
        display_step = min(global_step, self.args.max_steps)

        save_dir = Path(self.trainer.default_root_dir) / "validation" / f"step_{display_step:06d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"{name}.mp4"
        save_video(pred_video, str(save_path), fps=self.args.fps)
                    
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe.denoising_model().train()

        return metric_logs

    def on_validation_epoch_end(self) -> None:
        # 只在主进程输出一次验证集平均指标
        if self.trainer.sanity_checking:
            return
    
        if not self.trainer.is_global_zero:
            return
        
        metrics = {}
        for k, v in self.trainer.callback_metrics.items():
            if isinstance(k, str) and k.startswith("val/"):
                if torch.is_tensor(v):
                    metrics[k] = float(v.detach().cpu())
                elif isinstance(v, (int, float)):
                    metrics[k] = float(v)
        
        if not metrics:
            return

        global_step = int(getattr(self.trainer, "global_step", 0) or 0)
        metric_str = ", ".join(f"{k}: {v:.4f}" for k, v in sorted(metrics.items()))
        py_logger.info(f"[Validation Summary][step={global_step}] {metric_str}")

    def configure_optimizers(self):
        trainable_params = [p for p in self.pipe.dit.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )
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
        """Reject full-state resume across TC/TE stage boundaries."""
        source_args = checkpoint.get("wan_stvsr_args", {})
        source_stage = source_args.get("train_stage") if isinstance(source_args, dict) else None
        if source_stage != self.args.train_stage:
            raise ValueError(
                "--resume_from_checkpoint requires the same training stage: "
                f"current={self.args.train_stage!r}, checkpoint={source_stage!r}"
            )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Keep only DiT weights in Lightning state_dict while preserving trainer states."""
        state_dict = checkpoint.get("state_dict", {})
        checkpoint["state_dict"] = {k: v for k, v in state_dict.items() if k.startswith("pipe.dit.")}
        checkpoint["wan_stvsr_args"] = vars(self.args)

def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, Optional[DataLoader]]:
    if args.train_stage == "tc":
        train_dataset = SpatioTemporalRealSRDataset(
            data_root=args.train_data_root,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            spatial_scale=args.spatial_scale,
            temporal_scale=args.temporal_scale,
            degradation_config=args.degradation_config,
            inter_frame_margin=args.inter_frame_margin,
        )
    else:
        train_dataset = SpatioTemporalRealSRImageVideoDataset(
            video_data_root=args.video_data_root,
            image_data_root=args.image_data_root,
            image_sample_ratio=args.image_sample_ratio,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            spatial_scale=args.spatial_scale,
            temporal_scale=args.temporal_scale,
            degradation_config=args.degradation_config,
            inter_frame_margin=args.inter_frame_margin,
            seed=args.seed,
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

    val_loader = None
    if args.train_stage == "te" and args.val_lq_root and args.val_hq_root:
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


def setup_file_logger(output_path: str) -> None:
    log_dir = Path(output_path) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 避免 resume 或多次初始化时重复添加 handler
    if getattr(root_logger, "_wanstvsr_logger_initialized", False):
        return

    formatter = logging.Formatter(LOG_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    root_logger.handlers.clear()
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    root_logger._wanstvsr_logger_initialized = True

    py_logger.info(f"Logging to file: {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WanSTVSR with online raw-video VAE encoding.")
    parser.add_argument("--train_stage", type=str, required=True, choices=["tc", "te"])

    # Paths
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--train_data_root", type=str, default=None)
    parser.add_argument("--video_data_root", type=str, default=None)
    parser.add_argument("--image_data_root", type=str, default=None)
    parser.add_argument("--val_lq_root", type=str, default=None)
    parser.add_argument("--val_hq_root", type=str, default=None)
    parser.add_argument("--degradation_config", type=str, required=True)
    parser.add_argument("--dit_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--text_encoder_path", type=str, default=None)
    parser.add_argument("--empty_prompt_embedding_path", type=str, default=None)
    parser.add_argument("--init_from_checkpoint", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # Dataset
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--spatial_scale", type=int, default=4)
    parser.add_argument("--temporal_scale", type=int, default=2)
    parser.add_argument("--image_sample_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inter_frame_margin", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--val_dataloader_num_workers", type=int, default=1)
    parser.add_argument("--max_val_samples", type=int, default=10)
    parser.add_argument("--prompt", type=str, default="")

    # Training
    parser.add_argument("--gpu_num", type=int, default=1, help="Number of GPUs used for training.")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size_per_gpu", type=int, default=1, help="Image batch size per GPU.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="The number of batches in gradient accumulation.")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
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

    # OSDEnhancer-style TC/TE loss weights.
    parser.add_argument("--lambda_res", type=float, default=1.0)
    parser.add_argument("--lambda_dists", type=float, default=1.0)
    parser.add_argument("--lambda_warp", type=float, default=0.05)
    parser.add_argument("--lambda_nqa", type=float, default=0.05)
    parser.add_argument("--nqa_chunk_size", type=int, default=4)

    # RAFT is initialized only by TE.
    parser.add_argument("--raft_ckpt_path", type=str, default="./utils/RAFT/raft-things.pth")

    # Validation/logging/checkpoints
    parser.add_argument("--val_check_interval", type=int, default=500, help="The number of steps between validation checks.")
    parser.add_argument("--statistic_frequency", type=int, default=500, help="The number of steps between memory statistics logging.")
    parser.add_argument("--checkpoint_every_n_train_steps", type=int, default=1000)
    parser.add_argument("--metrics", type=str, default="psnr,ssim,lpips,dists,clipiqa")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--video_quality", type=int, default=9)
    parser.add_argument("--tensorboard_name", type=str, default="WanSTVSRv1")

    args = parser.parse_args()
    if args.init_from_checkpoint and args.resume_from_checkpoint:
        raise ValueError("--init_from_checkpoint and --resume_from_checkpoint are mutually exclusive")
    if (args.height, args.width) != (320, 640):
        raise ValueError(f"TC/TE training requires 320x640, got {args.height}x{args.width}")
    if args.nqa_chunk_size <= 0:
        raise ValueError("--nqa_chunk_size must be positive")

    if args.train_stage == "tc":
        if not args.train_data_root:
            raise ValueError("--train_data_root is required for TC")
        if args.num_frames != 13:
            raise ValueError(f"TC requires --num_frames 13, got {args.num_frames}")
    else:
        if not args.video_data_root or not args.image_data_root:
            raise ValueError("--video_data_root and --image_data_root are required for TE")
        if args.num_frames != 9:
            raise ValueError(f"TE requires --num_frames 9, got {args.num_frames}")
        if not args.init_from_checkpoint and not args.resume_from_checkpoint:
            raise ValueError("TE requires Stage-1 --init_from_checkpoint or same-stage --resume_from_checkpoint")
        if not args.raft_ckpt_path:
            raise ValueError("--raft_ckpt_path is required for TE")
        if pyiqa is None:
            raise ImportError("pyiqa is required for TE")

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_path) / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    return args


def train(args: argparse.Namespace) -> None:
    # setup_file_logger(args.output_path)
    pl.seed_everything(args.seed, workers=True)

    train_loader, val_loader = build_dataloaders(args)
    dataset_size = len(train_loader.dataset)
    raw_train_batches = len(train_loader)
    
    global_batch_size = args.gpu_num * args.batch_size_per_gpu
    effective_batch_size = (global_batch_size * args.accumulate_grad_batches)

    usable_samples = (dataset_size // effective_batch_size) * effective_batch_size
    limit_train_batches = usable_samples // global_batch_size

    if limit_train_batches <= 0:
        raise ValueError(
            f"Invalid limit_train_batches={limit_train_batches}. "
            f"dataset_size={dataset_size}, "
            f"gpu_num={args.gpu_num}, "
            f"batch_size_per_gpu={args.batch_size_per_gpu}, "
            f"accumulate_grad_batches={args.accumulate_grad_batches}, "
            f"effective_batch_size={effective_batch_size}."
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

    model = LightningModelForTrain(args)

    tensor_logger = TensorBoardLogger(
        save_dir=str(Path(args.output_path) / "tensorboardlog"),
        name=args.tensorboard_name,
        default_hp_metric=False,
    )
    logger = [tensor_logger]

    val_check_interval_batches = None
    if val_loader is not None and args.val_check_interval > 0:
        val_check_interval_batches = args.val_check_interval * args.accumulate_grad_batches

    trainer = pl.Trainer(
        max_steps=args.max_steps,
        accelerator="gpu",
        devices=args.gpu_num,
        strategy=args.training_strategy,
        precision="bf16",
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1, every_n_train_steps=args.checkpoint_every_n_train_steps)],
        logger=logger,
        log_every_n_steps=1,
        val_check_interval=val_check_interval_batches,
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

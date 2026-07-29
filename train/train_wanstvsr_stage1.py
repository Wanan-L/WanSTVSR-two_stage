#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-1 TC training for WanSTVSR.

TC training only uses latent reconstruction and latent temporal residual
consistency losses. The original paired-video validation path is retained and
computes PSNR, SSIM, LPIPS, DISTS, and CLIPIQA without changing the TC loss.
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
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from diffsynth import ModelManager
from diffsynth.pipelines.wan_stvsr import WanSTVSRPipeline
from dataset.spatio_temporal_real_sr_dataset import SpatioTemporalRealSRDataset
from dataset.utils import (
    bcthw_to_fchw_01,
    prepare_input_tensor,
    read_video_as_tensor,
    save_video,
    scan_video_or_frame_dirs,
)
from utils.memory_utils import (
    free_memory,
    get_memory_statistics,
    reset_peak_memory_stats,
)
from utils.metric_utils import evaluate_video_metrics

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stdout)
py_logger = logging.getLogger(__name__)

STAGE_NAME = "stage1-tc"


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


class LightningModelForStage1(pl.LightningModule):
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

        self.prompt_text = args.prompt
        self.context: Optional[torch.Tensor] = None     # 使用统一的 context
        self.metric_models: Dict[str, Any] = {}
        self._last_memory_log_step = -1         # 避免梯度累积时，同一个 global_step 被 on_train_batch_end 重复记录

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

    def setup(self, stage: Optional[str] = None) -> None:
        if self.context is None:
            self.context = self._load_or_encode_context()
        # if stage in ("fit", "validate", "test", None):
        #     self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize validation-only IQA metrics; these are not losses."""
        if self.metric_models:
            return
        
        metric_names = [name.strip().lower() for name in self.args.metrics.split(",") if name.strip()]
        for name in metric_names:
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

    def forward_one_step(self, z_l: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable one-step DiT forward used by training."""
        timestep = torch.full((1,), self.args.fixed_timestep, dtype=self.torch_dtype, device=z_l.device,)

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

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        self.pipe.device = self.device

        hq_video = batch["hq_video"].to(device=self.device, dtype=self.torch_dtype)
        lq_video = batch["lq_video"].to(device=self.device, dtype=self.torch_dtype)

        z_gt, _ = self.pad_and_encode(hq_video)
        z_l, _ = self.pad_and_encode(lq_video)
        context = self.get_context(hq_video.shape[0])
        z_out, _ = self.forward_one_step(z_l, context)

        if z_out.shape != z_gt.shape:
            raise ValueError(f"Latent shape mismatch: z_out={tuple(z_out.shape)}, z_gt={tuple(z_gt.shape)}")
        if z_out.shape[2] <= 1:
            raise ValueError(f"TC latent temporal length must be greater than 1, got {z_out.shape[2]}.")

        if not self._shape_logged:
            py_logger.info(
                "Training shapes: " "hq_video=%s, lq_video=%s, " "z_gt=%s, z_l=%s",
                tuple(hq_video.shape), tuple(lq_video.shape), tuple(z_gt.shape), tuple(z_l.shape),
            )
            self._shape_logged = True

        loss_mse = F.mse_loss(z_out.float(), z_gt.float())
        delta_out = z_out[:, :, 1:] - z_out[:, :, :-1]
        delta_gt = z_gt[:, :, 1:] - z_gt[:, :, :-1]
        loss_res = F.l1_loss(delta_out.float(), delta_gt.float(), reduction="mean")
        loss = loss_mse + self.args.lambda_res * loss_res

        self._log_training(
            loss,
            {
                "train/loss_mse": loss_mse.detach(),
                "train/loss_res": loss_res.detach(),
                "train/latent_length": z_out.new_tensor(float(z_out.shape[2])),
            },
            hq_video.shape[0],
        )
        return loss

    def on_train_start(self) -> None:
        self.pipe.vae.eval()
        self.pipe.denoising_model().train()

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
        self._log_memory("Before_train", "Memory before training start")
        reset_peak_memory_stats(self.device)

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        freq = getattr(self.args, "statistic_frequency", 0)
        step = int(getattr(self.trainer, "global_step", 0) or 0)
        if freq > 0 and step > 0 and step % freq == 0 and step != self._last_memory_log_step:
            self._last_memory_log_step = step
            self._log_memory("After_iter", f"Memory after optimizer step {step} (frequency: {freq})")

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """在每次优化器更新参数之前，统计当前 DiT 参数的梯度大小"""
        total_norm_sq = 0.0         # 总梯度范数平方
        max_parameter_norm = 0.0    # 最大单参数梯度范数

        for parameter in self.pipe.dit.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                parameter_norm = parameter.grad.detach().float().norm(2).item()     # 单个参数张量的梯度 norm
                total_norm_sq += parameter_norm**2
                max_parameter_norm = max(max_parameter_norm, parameter_norm)

        self.log_dict(
            {
                "train/grad_norm": total_norm_sq**0.5,
                "train/grad_norm_max_param": max_parameter_norm,
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
        """Run the unchanged inference-style validation path for stage1 checkpoints."""
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
        if self.trainer.sanity_checking:
            return
        if not self.metric_models:
            self._init_metrics()
        else:
            # 上一次验证结束后指标被移到了 CPU，本次验证开始前重新移回当前 GPU。
            for metric in self.metric_models.values():
                metric.to(self.device)
                metric.eval()

        self._log_memory("Before_val", "Memory before validation start")

    def on_validation_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        
        self._log_memory("After_val", "Memory after validation end")
        for metric in self.metric_models.values():   # 验证结束后将指标移回 CPU，释放训练阶段显存。
            metric.to("cpu")
            metric.eval()
        free_memory()
        self._log_memory("After_val_free", "Memory after moving metrics to CPU")
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
        trainable_parameters = [
            parameter
            for parameter in self.pipe.dit.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
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
        source_stage = checkpoint_stage(checkpoint)
        if source_stage != STAGE_NAME:
            raise ValueError(
                "Stage1 --resume_from_checkpoint requires Stage1 checkpoint, "
                f"but checkpoint stage is {source_stage!r}."
            )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Keep only DiT weights in Lightning state_dict while preserving trainer states."""
        state_dict = checkpoint.get("state_dict", {})
        checkpoint["state_dict"] = {
            key: value
            for key, value in state_dict.items()
            if key.startswith("pipe.dit.")
        }
        checkpoint["wan_stvsr_stage"] = STAGE_NAME
        checkpoint["wan_stvsr_args"] = vars(self.args)


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, Optional[DataLoader]]:
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
    parser = argparse.ArgumentParser(description="WanSTVSR Stage-1 TC latent training.")

    # Paths
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--train_data_root", type=str, required=True)
    parser.add_argument("--val_lq_root", type=str, default=None)
    parser.add_argument("--val_hq_root", type=str, default=None)
    parser.add_argument("--degradation_config", type=str, required=True)
    parser.add_argument("--dit_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--text_encoder_path", type=str, default=None)
    parser.add_argument("--empty_prompt_embedding_path", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # Dataset
    parser.add_argument("--num_frames", type=int, default=33)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--spatial_scale", type=int, default=4)
    parser.add_argument("--temporal_scale", type=int, default=2)
    parser.add_argument("--inter_frame_margin", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--val_dataloader_num_workers", type=int, default=1)
    parser.add_argument("--max_val_samples", type=int, default=10)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)

    # Training
    parser.add_argument("--gpu_num", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
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
    parser.add_argument("--lambda_res", type=float, default=1.0)
    parser.add_argument("--val_check_interval", type=int, default=500)
    parser.add_argument("--statistic_frequency", type=int, default=500)
    parser.add_argument("--checkpoint_every_n_train_steps", type=int, default=1000)
    parser.add_argument("--metrics", type=str, default="psnr,ssim,lpips,dists,clipiqa")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--video_quality", type=int, default=9)
    parser.add_argument("--tensorboard_name", type=str, default="WanSTVSRv1")

    args = parser.parse_args()
    if args.num_frames < 2:
        raise ValueError("--num_frames must be at least 2 for residual loss.")
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

    model = LightningModelForStage1(args)
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
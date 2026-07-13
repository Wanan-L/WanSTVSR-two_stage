#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WanSTVSR training script.

Design:
  - STCDiT-style PyTorch Lightning training skeleton.
  - Raw-video online dataset, no precomputed latent .pth.
  - VAE/text encoder/RAFT/metric networks are frozen.
  - Only full DiT parameters are optimized.
  - Loss = latent MSE + reconstruction MSE + LPIPS + temporal consistency.
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

        self.prompt_text = args.prompt
        self.context: Optional[torch.Tensor] = None     # 使用统一的 context
        self.lpips_loss = None
        self.metric_models: Dict[str, Any] = {}
        self.raft: Optional[Any] = None

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

    def setup(self, stage: Optional[str] = None) -> None:
        """"准备 prompt embedding，初始化训练 loss 和验证 metrics"""
        if self.context is None:
            self.context = self._load_or_encode_context()
        if stage in ("fit", None):
            self._init_losses()
        if stage in ("fit", "validate", "test", None):
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
        if self.args.lambda_perc > 0:
            if pyiqa is None:
                raise ImportError("pyiqa is required when --lambda_perc > 0")
            self.lpips_loss = pyiqa.create_metric("lpips", device=self.device, as_loss=True).eval()
            for p in self.lpips_loss.parameters():
                p.requires_grad_(False)

        if self.args.use_consis_loss and self.args.lambda_consis > 0:
            self.raft = RAFT_bi(self.args.raft_ckpt_path, device=self.device)
            if hasattr(self.raft, "requires_grad_"):
                self.raft.requires_grad_(False)
            if hasattr(self.raft, "eval"):
                self.raft.eval()

    def _init_metrics(self) -> None:
        if pyiqa is None:
            raise ImportError("pyiqa is required for validation metrics")
        for name in [m.strip().lower() for m in self.args.metrics.split(",") if m.strip()]:
            if name in self.metric_models:
                continue
            self.metric_models[name] = pyiqa.create_metric(name, device=self.device).eval()
            if hasattr(self.metric_models[name], "parameters"):
                for p in self.metric_models[name].parameters():
                    p.requires_grad_(False)

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
        self._log_memory("Before_train", "Memory before training start")
        reset_peak_memory_stats(self.device)

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

    def decode_with_grad(self, latent: torch.Tensor, pad_info: Tuple[int, int, int]) -> torch.Tensor:
        # VAE params are frozen, but decoder graph must stay for image-domain losses.
        pad_h, pad_w, pad_t = pad_info
        video = self._first_tensor(self.pipe.decode_video(latent, **self.tiler_kwargs))
        video = self.pipe.unpad_temporal(video, pad_t)
        video = self.pipe.unpad_spatial(video, pad_h, pad_w)
        return video.clamp(-1, 1).contiguous()

    def forward_one_step(self, z_l: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Trainable one-step WanSTVSR forward.
        It directly calls WanModel.forward() so gradient checkpointing/offload can be controlled through WanModel's forward arguments.
        """
        timestep = torch.full((z_l.shape[0],), self.args.fixed_timestep, dtype=self.torch_dtype, device=z_l.device,)

        z_in = z_l
        if self.args.noise_step > 0:
            noise = torch.randn_like(z_l)
            noise_timestep = torch.full((z_l.shape[0],), self.args.noise_step, dtype=self.torch_dtype, device=self.device)
            z_in = self.pipe.scheduler.add_noise(z_l, noise, timestep=noise_timestep)

        noise_pred = self.pipe.denoising_model()(
            z_in,
            timestep=timestep,
            context=context,
            **self.pipe.prepare_extra_input(z_in),
            use_gradient_checkpointing=self.args.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.args.use_gradient_checkpointing_offload,
        )
        z_st = z_in - noise_pred
        return z_st, noise_pred

    def compute_perceptual_loss(self, pred_video: torch.Tensor, target_video: torch.Tensor) -> torch.Tensor:
        # pred_video / target_video: [B, C, T, H, W]
        if self.lpips_loss is None or self.args.lambda_perc <= 0:
            return pred_video.new_tensor(0.0)
            
        pred_01 = ((pred_video.float() + 1.0) * 0.5).clamp(0, 1)
        target_01 = ((target_video.float() + 1.0) * 0.5).clamp(0, 1)

        b, c, t, h, w = pred_01.shape
        pred_4d = pred_01.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).contiguous()
        target_4d = target_01.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).contiguous()

        self.lpips_loss.float()

        with torch.autocast(device_type=pred_video.device.type, enabled=False):
            loss = self.lpips_loss(pred_4d.float(), target_4d.float()).mean()

        return loss.float()

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

    def compute_temporal_consistency_loss(self, pred_video: torch.Tensor, target_video: torch.Tensor) -> torch.Tensor:
        if not self.args.use_consis_loss or self.args.lambda_consis <= 0:
            return pred_video.new_tensor(0.0)
        if self.raft is None:
            raise RuntimeError("RAFT is not initialized. Use --use_consis_loss and provide --raft_ckpt_path.")

        pred_01 = ((pred_video.float() + 1.0) * 0.5).clamp(0, 1)      # [-1,1] -> [0,1]
        target_01 = ((target_video.float() + 1.0) * 0.5).clamp(0, 1)
        b, c, t, h, w = pred_01.shape
        if t < 2:
            return pred_video.new_tensor(0.0)     # 帧数小于 2 时，无法计算光流

        # RAFT 只用于估计 target 上的光流，不需要梯度
        self.raft.eval()
        self.raft.requires_grad_(False)

        with torch.no_grad():
            self.raft.float()
            with torch.autocast(device_type=pred_video.device.type, enabled=False):
                raft_out = self.raft(target_01.float())

            if isinstance(raft_out, dict):
                flows_forward = None
                flows_backward = None     # 在 target video 上估计前向/后向光流

                for key in ["flows_forward", "forward", "flow_forward"]:
                    if key in raft_out and raft_out[key] is not None:
                        flows_forward = raft_out[key]
                        break

                for key in ["flows_backward", "backward", "flow_backward"]:
                    if key in raft_out and raft_out[key] is not None:
                        flows_backward = raft_out[key]
                        break
            elif isinstance(raft_out, (list, tuple)) and len(raft_out) == 2:
                flows_forward, flows_backward = raft_out

            else:
                raise RuntimeError(
                    f"Unsupported RAFT_bi output type: {type(raft_out)}. "
                    "Expected dict or tuple/list of (flows_forward, flows_backward)."
                )

            if flows_forward is None or flows_backward is None:
                raise RuntimeError("RAFT_bi must return flows_forward and flows_backward")
            if self.args.swap_flow_directions:      # 交换光流方向
                flows_forward, flows_backward = flows_backward, flows_forward

            expected_pairs = t - 1
            flows_forward = self._resize_flow_to_video(flows_forward.to(device=pred_01.device, dtype=torch.float32), h, w, expected_pairs=expected_pairs, name="flows_forward")
            flows_backward = self._resize_flow_to_video(flows_backward.to(device=pred_01.device, dtype=torch.float32), h, w, expected_pairs=expected_pairs, name="flows_backward")

        losses: List[torch.Tensor] = []
        eps = 1e-6
        for i in range(t - 1):      # 逐帧按 pair 计算光流损失
            # 假设有 4 帧：0, 1, 2, 3
            # 代码按 pair 算：
            # pair (0,1): 1 -> 0, 0 -> 1; 
            # pair (1,2): 2 -> 1, 1 -> 2; 
            # pair (2,3): 3 -> 2, 2 -> 3
            # DiffST 按中心帧算：
            # center 1: 2 -> 1, 0 -> 1; 
            # center 2: 3 -> 2, 1 -> 2
            # 区别在于这里约束了边界帧，本质上是相邻帧双向 warp consistency

            # 用 forward flow 把 pred_{i+1} warp 到 pred_i
            flow_f = flows_forward[:, :, i]
            warped_next = flow_warp(pred_01[:, :, i + 1], flow_f.permute(0, 2, 3, 1))
            diff_next = (warped_next - pred_01[:, :, i]).abs()
            if self.args.use_flow_mask:       # 使用光流 mask（默认不使用）
                mask_f = fbConsistencyCheck(flow_f, flows_backward[:, :, i])
                mask_f = mask_f.to(device=diff_next.device, dtype=diff_next.dtype)
                losses.append((diff_next * mask_f).sum() / (mask_f.sum() * c + eps))
            else:
                losses.append(diff_next.mean())

            # 用 backward flow 把 pred_i warp 到 pred_{i+1}
            flow_b = flows_backward[:, :, i]
            warped_prev = flow_warp(pred_01[:, :, i], flow_b.permute(0, 2, 3, 1))
            diff_prev = (warped_prev - pred_01[:, :, i + 1]).abs()
            if self.args.use_flow_mask:
                mask_b = fbConsistencyCheck(flow_b, flow_f)
                mask_b = mask_b.to(device=diff_prev.device, dtype=diff_prev.dtype)
                losses.append((diff_prev * mask_b).sum() / (mask_b.sum() * c + eps))
            else:
                losses.append(diff_prev.mean())

        return torch.stack(losses).mean().float()

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        self.pipe.device = self.device
        hq_video = batch["hq_video"].to(device=self.device, dtype=self.torch_dtype)
        lq_video = batch["lq_video"].to(device=self.device, dtype=self.torch_dtype)
        batch_size = hq_video.shape[0]

        z_h, _ = self.pad_and_encode(hq_video)
        z_l, pad_info = self.pad_and_encode(lq_video)
        context = self.get_context(batch_size)
        z_st, _ = self.forward_one_step(z_l, context)

        if z_st.shape != z_h.shape:
            raise ValueError(f"Latent shape mismatch: z_st={z_st.shape}, z_h={z_h.shape}")

        loss_latent = F.mse_loss(z_st.float(), z_h.float())
        pred_video = self.decode_with_grad(z_st, pad_info)

        if pred_video.shape != hq_video.shape:
            raise ValueError(f"Video shape mismatch: pred={pred_video.shape}, hq={hq_video.shape}")
        
        # 不使用切片静默裁剪，否则可能掩盖 temporal_scale/spatial_scale 或 padding 逻辑错误
        # target_video = hq_video[:, :, : pred_video.shape[2], : pred_video.shape[3], : pred_video.shape[4]]
        target_video = hq_video

        loss_rec = F.mse_loss(pred_video.float(), target_video.float())
        loss_perc = self.compute_perceptual_loss(pred_video, target_video)
        loss_consis = self.compute_temporal_consistency_loss(pred_video, target_video)

        loss = (
            self.args.lambda_latent * loss_latent
            + self.args.lambda_rec * loss_rec
            + self.args.lambda_perc * loss_perc
            + self.args.lambda_consis * loss_consis
        )

        logs = {
            "train/loss_latent": loss_latent.detach(),
            "train/loss_rec": loss_rec.detach(),
            "train/loss_perc": loss_perc.detach(),
            "train/loss_consis": loss_consis.detach(),
        }

        completed_step = int(getattr(self.trainer, "global_step", 0) or 0)
        display_step = min(completed_step + 1, self.args.max_steps)
        progress_pct = 100.0 * display_step / max(float(self.args.max_steps), 1.0)  # 显示训练进度

        self.log("Progress",progress_pct, prog_bar=True, logger=False, on_step=True, on_epoch=False, sync_dist=False, rank_zero_only=True)
        self.log("train/loss", loss.detach(), prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True,)
        self.log_dict(logs, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=True)  # loss 多卡取平均
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=False, rank_zero_only=True)


        return loss

    @torch.no_grad()
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

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Keep only DiT weights in Lightning state_dict while preserving trainer states."""
        state_dict = checkpoint.get("state_dict", {})
        checkpoint["state_dict"] = {k: v for k, v in state_dict.items() if k.startswith("pipe.dit.")}
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

    val_loader = None
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
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--spatial_scale", type=int, default=4)
    parser.add_argument("--temporal_scale", type=int, default=2)
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

    # Loss weights. Defaults are 1:1:1:0.1.
    parser.add_argument("--lambda_latent", type=float, default=1.0)
    parser.add_argument("--lambda_rec", type=float, default=1.0)
    parser.add_argument("--lambda_perc", type=float, default=1.0)
    parser.add_argument("--lambda_consis", type=float, default=0.1)

    # RAFT / temporal consistency
    parser.add_argument("--use_consis_loss", action="store_true", default=False)
    parser.add_argument("--raft_ckpt_path", type=str, default="./utils/RAFT/raft-things.pth")
    parser.add_argument("--use_flow_mask", action="store_true", default=False)
    parser.add_argument("--swap_flow_directions", action="store_true", default=False)

    # Validation/logging/checkpoints
    parser.add_argument("--val_check_interval", type=int, default=500, help="The number of steps between validation checks.")
    parser.add_argument("--statistic_frequency", type=int, default=500, help="The number of steps between memory statistics logging.")
    parser.add_argument("--checkpoint_every_n_train_steps", type=int, default=1000)
    parser.add_argument("--metrics", type=str, default="psnr,ssim,lpips,dists,clipiqa")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--video_quality", type=int, default=9)
    parser.add_argument("--tensorboard_name", type=str, default="WanSTVSRv1")

    args = parser.parse_args()
    if args.use_consis_loss and args.lambda_consis > 0 and not args.raft_ckpt_path:
        raise ValueError("--raft_ckpt_path is required when --use_consis_loss is enabled")
    if args.lambda_perc > 0 and pyiqa is None:
        raise ImportError("pyiqa is required when --lambda_perc > 0")

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_path) / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    return args


def train(args: argparse.Namespace) -> None:
    # setup_file_logger(args.output_path)

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

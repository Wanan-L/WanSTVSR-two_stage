import types
import os
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import ModelManager
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from .base import BasePipeline
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear

# -----------------------------
# 基础工具：ADAIN 所需的统计量（保留以备需要；管线默认用 wavelet）
# -----------------------------
def _calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    assert feat.dim() == 4, 'feat 必须是 (N, C, H, W)'
    N, C = feat.shape[:2]
    var = feat.view(N, C, -1).var(dim=2, unbiased=False) + eps
    std = var.sqrt().view(N, C, 1, 1)
    mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std


def _adain(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    assert content_feat.shape[:2] == style_feat.shape[:2], "ADAIN: N、C 必须匹配"
    size = content_feat.size()
    style_mean, style_std = _calc_mean_std(style_feat)
    content_mean, content_std = _calc_mean_std(content_feat)
    normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


# -----------------------------
# 小波式模糊与分解/重构（ColorCorrector 用）
# -----------------------------
def _make_gaussian3x3_kernel(dtype, device) -> torch.Tensor:
    vals = [
        [0.0625, 0.125, 0.0625],
        [0.125,  0.25,  0.125 ],
        [0.0625, 0.125, 0.0625],
    ]
    return torch.tensor(vals, dtype=dtype, device=device)


def _wavelet_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    assert x.dim() == 4, 'x 必须是 (N, C, H, W)'
    N, C, H, W = x.shape
    base = _make_gaussian3x3_kernel(x.dtype, x.device)
    weight = base.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    pad = radius
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='replicate')
    out = F.conv2d(x_pad, weight, bias=None, stride=1, padding=0, dilation=radius, groups=C)
    return out


def _wavelet_decompose(x: torch.Tensor, levels: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 4:
        raise ValueError(f"Expected [N, C, H, W], got {tuple(x.shape)}")
    high = torch.zeros_like(x)
    low = x
    for i in range(levels):
        blurred = _wavelet_blur(low, radius=2**i)
        high = high + (low - blurred)
        low = blurred
    return high, low


def _wavelet_reconstruct(content: torch.Tensor, style: torch.Tensor, levels: int = 5) -> torch.Tensor:
    content_high, _ = _wavelet_decompose(content, levels=levels)
    _, style_low = _wavelet_decompose(style, levels=levels)
    return content_high + style_low


# -----------------------------
# 无状态颜色矫正模块（视频友好，默认 wavelet）
# -----------------------------
class TorchColorCorrectorWavelet(nn.Module):
    def __init__(self, levels: int = 5):
        super().__init__()
        self.levels = levels

    @staticmethod
    def _flatten_time(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        if x.dim() != 5:
            raise ValueError(f"Expected [B, C, T, H, W], got {tuple(x.shape)}")
        b, c, t, h, w = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), b, t

    @staticmethod
    def _unflatten_time(x: torch.Tensor, b: int, t: int) -> torch.Tensor:
        bt, c, h, w = x.shape
        if bt != b * t:
            raise ValueError(f"Expected first dim {b * t}, got {bt}")
        return x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

    def forward(
        self,
        hq_image: torch.Tensor,  # (B, C, f, H, W)
        lq_image: torch.Tensor,  # (B, C, f, H, W)
        clip_range: Tuple[float, float] = (-1.0, 1.0),
        method: Literal["wavelet", "adain"] = "wavelet",
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if hq_image.shape != lq_image.shape:
            raise ValueError(f"Color correction expects same shape, got {hq_image.shape} and {lq_image.shape}")
        if hq_image.dim() != 5 or hq_image.shape[1] != 3:
            raise ValueError(f"Expected [B, 3, T, H, W], got {tuple(hq_image.shape)}")

        _, _, total_t, _, _ = hq_image.shape
        if chunk_size is None or chunk_size >= total_t:
            hq_4d, b, t = self._flatten_time(hq_image)
            lq_4d, _, _ = self._flatten_time(lq_image)
            out_4d = self._correct_4d(hq_4d, lq_4d, method=method)
            return self._unflatten_time(out_4d.clamp(*clip_range), b, t)

        chunks = []
        for start in range(0, total_t, chunk_size):
            end = min(start + chunk_size, total_t)
            hq_4d, b, t = self._flatten_time(hq_image[:, :, start:end])
            lq_4d, _, _ = self._flatten_time(lq_image[:, :, start:end])
            out_4d = self._correct_4d(hq_4d, lq_4d, method=method)
            chunks.append(self._unflatten_time(out_4d.clamp(*clip_range), b, t))
        return torch.cat(chunks, dim=2)

    def _correct_4d(
        self,
        hq_image: torch.Tensor,
        lq_image: torch.Tensor,
        method: Literal["wavelet", "adain"],
    ) -> torch.Tensor:
        if method == "wavelet":
            return _wavelet_reconstruct(hq_image, lq_image, levels=self.levels)
        if method == "adain":
            return _adain(hq_image, lq_image)
        raise ValueError(f"Unknown color correction method: {method}")



class WanSTVSRPipeline(BasePipeline):
    """Wan2.1-T2V baseline pipeline for one-step video spatio-temporal SR.

    The input LQ video is expected to be pre-aligned to the target GT shape
    before entering the pipeline. No LQ upsampling or extra LQ condition module
    is introduced here.
    """

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ["dit", "vae", "text_encoder"]
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.use_unified_sequence_parallel = False
        self.ColorCorrector = TorchColorCorrectorWavelet(levels=5)

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        # 仅管理 dit / vae
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )

        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        self.enable_cpu_offload()

    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")

    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None, use_usp=False):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = WanSTVSRPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

            for block in pipe.dit.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            pipe.dit.forward = types.MethodType(usp_dit_forward, pipe.dit)
            pipe.sp_size = get_sequence_parallel_world_size()
            pipe.use_unified_sequence_parallel = True
        return pipe
    
    def denoising_model(self):
        return self.dit

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive, device=self.device)
        return {"context": prompt_emb}  # 返回 list
    
    def prepare_unified_sequence_parallel(self):
        return {"use_unified_sequence_parallel": self.use_unified_sequence_parallel}

    def prepare_extra_input(self, latents=None):
        return {}

    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames
    
    
    @torch.no_grad()
    def test(
        self,
        prompt="",
        negative_prompt="",
        lq_video=None,      # tensor
        prompt_emb_posi=None,
        prompt_emb_nega=None,
        denoising_strength=1.0,
        cfg_scale=1.0,
        num_inference_steps=1,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(34, 34),
        tile_stride=(18, 16),
        fixed_timestep=799,
        noise_step=0,     # 噪声步长, 0 表示不添加噪声
        color_fix = True,
    ):
        if lq_video is None:
            raise ValueError("lq_video must be provided.")
        if cfg_scale < 0:
            raise ValueError(f"cfg_scale must be non-negative, got {cfg_scale}.")

        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Encode input video to latents
        self.load_models_to_device(['vae'])

        lq_video = lq_video.to(dtype=self.torch_dtype, device=self.device).contiguous()

        lq_video_pad, pad_h, pad_w = self.pad_spatial(lq_video)
        lq_video_pad, pad_t = self.pad_temporal(lq_video_pad)
        
        z_l = self.encode_video(lq_video_pad, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
        z_l = z_l.detach()      # 防止梯度传递

        # 固定 timestep
        timestep = torch.full((z_l.shape[0],), fixed_timestep, dtype=self.torch_dtype, device=self.device)

        z_in = z_l
        if noise_step > 0:
            noise = torch.randn_like(z_l)
            noise_timestep = torch.full((z_l.shape[0],), noise_step, dtype=self.torch_dtype, device=self.device)
            z_in = self.scheduler.add_noise(z_l, noise, timestep=noise_timestep)

        # Encode the positive prompt. A precomputed embedding takes precedence
        # over prompt text, which is useful for empty-prompt training/inference.
        if prompt_emb_posi is None:
            self.load_models_to_device(["text_encoder"])
            prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        else:
            if isinstance(prompt_emb_posi, torch.Tensor):
                prompt_emb_posi = {"context": prompt_emb_posi}
            prompt_emb_posi["context"] = prompt_emb_posi["context"].to(dtype=self.torch_dtype, device=self.device)

        # Negative prompt is only needed by classifier-free guidance.
        if cfg_scale != 1.0:
            if prompt_emb_nega is None:
                self.load_models_to_device(["text_encoder"])
                prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            else:
                if isinstance(prompt_emb_nega, torch.Tensor):
                    prompt_emb_nega = {"context": prompt_emb_nega}
            prompt_emb_nega["context"] = prompt_emb_nega["context"].to(dtype=self.torch_dtype, device=self.device)

        # Extra input
        extra_input = self.prepare_extra_input(z_in)

        # Unified Sequence Parallel
        usp_kwargs = self.prepare_unified_sequence_parallel()

        # Denoise
        self.load_models_to_device(["dit"])
        noise_pred_posi = model_fn_wan_video(self.dit, x=z_in, timestep=timestep, **prompt_emb_posi, **extra_input, **usp_kwargs)
        if cfg_scale != 1.0:
            noise_pred_nega = model_fn_wan_video(self.dit, x=z_in, timestep=timestep, **prompt_emb_nega, **extra_input, **usp_kwargs)
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        else:
            noise_pred = noise_pred_posi
        
        z_st = z_in - noise_pred

        # Decode latents to video
        self.load_models_to_device(["vae"])
        frames = self.decode_video(z_st, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
        frames = self.unpad_spatial(self.unpad_temporal(frames, pad_t), pad_h, pad_w)
        frames = frames.clamp(-1, 1).contiguous()

        # Color correction (wavelet)
        if color_fix:
            frames = self.ColorCorrector(
                frames,
                lq_video[:, :, :frames.shape[2], :, :],
                clip_range=(-1, 1),
                chunk_size=16,
                method='adain'
            )

        return frames


def model_fn_wan_video(
    dit: WanModel,
    x: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    use_unified_sequence_parallel: bool = False,
    **kwargs,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group

    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)


    x, (f, h, w) = dit.patchify(x)

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    for block in dit.blocks:
        x = block(x, context, t_mod, freqs)

    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
    x = dit.unpatchify(x, (f, h, w))
    return x

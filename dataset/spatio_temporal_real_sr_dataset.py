import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
from torchvision import transforms

from .random_degradations import (
    DegradationsWithShuffle,
    RandomBlur,
    RandomJPEGCompression,
    RandomNoise,
    RandomResize,
    RandomVideoCompression,
)
from .utils import paired_random_crop_video, random_crop_frames, read_video_frames, scan_video_or_frame_dirs, temporal_blending


class SpatioTemporalRealSRDataset(Dataset):
    """HQ-VSR dataset with RealBasicVSR-style spatial degradation and temporal downsampling.

    The data flow follows DOVE's `RealSRDataset.preprocess`:
      1. read more intermediate frames from the HQ video;
      2. crop an intermediate HQ clip;
      3. run the two-stage RealBasicVSR degradation pipeline on the cropped HQ clip;
      4. paired crop HQ/LQ to the final training resolution;
      5. uniformly keep every `temporal_scale`-th LQ frame;
      6. bilinearly upsample spatially and blending interpolate temporally to build `lq_video`.

    Returned HQ and LQ tensors are both in [C, T, H, W], normalized to [-1, 1] for Wan VAE input.
    `lq_video` is already upsampled back to the HQ spatial size and frame count.
    """

    def __init__(
        self,
        data_root: str | Path | None = None,
        num_frames: int = 17,
        height: int = 320,
        width: int = 640,
        spatial_scale: int = 4,
        temporal_scale: int = 2,
        degradation_config: str | Path | None = None,
        inter_frame_margin: int = 10,
        image_exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".webp"),
    ) -> None:
        super().__init__()
        if data_root is None:
            raise ValueError("data_root must be provided")
        if degradation_config is None:
            raise ValueError("degradation_config must be provided")
        
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.spatial_scale = spatial_scale
        self.temporal_scale = temporal_scale
        if self.height % self.spatial_scale != 0 or self.width % self.spatial_scale != 0:
            raise ValueError(
                "height and width must be divisible by spatial_scale, got "
                f"height={self.height}, width={self.width}, spatial_scale={self.spatial_scale}"
            )
        if (self.num_frames - 1) % self.temporal_scale != 0:
            raise ValueError(
                "num_frames must satisfy (num_frames - 1) % temporal_scale == 0 "
                "so temporal downsampling keeps both endpoints, got "
                f"num_frames={self.num_frames}, temporal_scale={self.temporal_scale}"
            )
        
        self.image_exts = tuple(image_exts)
        self.videos = scan_video_or_frame_dirs(self.data_root)
        if not self.videos:
            raise FileNotFoundError(f"No video/frame samples found in {self.data_root}")

        self.inter_frames = self.num_frames + inter_frame_margin
        self.inter_height = math.ceil((self.height * 1.5) / 16) * 16
        self.inter_width = math.ceil((self.width * 1.5) / 16) * 16
        self.target_h = self.height // self.spatial_scale
        self.target_w = self.width // self.spatial_scale
        self.__frame_transform = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]) # -1, 1

        with open(degradation_config, "r", encoding="utf-8") as f:
            self.opt = yaml.safe_load(f)
        self.init_degradation(self.opt)

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.videos[index]
        # Current shape of frames: [T, C, H, W]
        hq_frames, lq_frames = self.preprocess(path)
        lq_video_resize = self.upsample_lq_video(lq_frames)

        # Convert to  [C, T, H, W]
        hq_video = self.video_transform(hq_frames).permute(1, 0, 2, 3).contiguous()
        lq_video = self.video_transform(lq_video_resize).permute(1, 0, 2, 3).contiguous()

        return {
            "hq_video": hq_video,
            "lq_video": lq_video,
            "path": str(path),
            "video_metadata": {
                "num_frames": self.num_frames,
                "height": self.height,
                "width": self.width,
            },
        }

    def preprocess(self, video_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_list = read_video_frames(video_path, self.inter_frames, self.image_exts)
        crop_frame_list = random_crop_frames(frame_list, self.inter_frames, self.inter_height, self.inter_width)
        inter_h, inter_w, _ = crop_frame_list[0].shape
        inter_target_h = inter_h // self.spatial_scale
        inter_target_w = inter_w // self.spatial_scale

        # 设置第二阶段退化的最终 resize 尺寸（下采样 spatial_scale 倍）
        if isinstance(self.degradation_with_shuffle.degradations[0],list):
            self.degradation_with_shuffle.degradations[0][0].params['target_size'] = (inter_target_h, inter_target_w)
        else:
            self.degradation_with_shuffle.degradations[1][0].params['target_size'] = (inter_target_h, inter_target_w)
        
        input_dict = dict(lqs=crop_frame_list)          # HQ clip [T, H, W, C]
        deg_frame_list = self.degrade(input_dict)['lqs']  # 空间退化

        hq_frame_list, lq_frame_list = paired_random_crop_video(
            crop_frame_list,
            deg_frame_list,
            self.num_frames,
            self.target_h,
            self.target_w,
            self.spatial_scale,
            str(video_path),
        )

        hq_video = torch.stack([self.to_tensor(frame) for frame in hq_frame_list], dim=0)   # [T, C, H, W]
        lq_video = torch.stack([self.to_tensor(frame) for frame in lq_frame_list], dim=0)

        if lq_video.ndim != 4:
            raise ValueError(f"Expected [T, C, H, W] LQ tensor, got shape {tuple(lq_video.shape)}")
        if lq_video.shape[0] != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} LQ frames, got {lq_video.shape[0]}")

        lq_video = lq_video[:: self.temporal_scale].contiguous()    # 时间退化

        expected_frames = (self.num_frames - 1) // self.temporal_scale + 1
        if lq_video.shape[0] != expected_frames:
            raise RuntimeError(f"Expected {expected_frames} low-fps frames, got {lq_video.shape[0]}")
        
        return hq_video, lq_video

    def upsample_lq_video(self, lq_video: torch.Tensor) -> torch.Tensor:
        """Upsample raw LQ [T, C, H, W] to the HQ frame count and spatial size."""
        t, c, h, w = lq_video.shape
        expected_h = self.height // self.spatial_scale
        expected_w = self.width // self.spatial_scale
        if h != expected_h or w != expected_w:
            raise ValueError(
                "Raw LQ size must match the configured spatial downsampling ratio, got "
                f"LQ=({h}, {w}), expected=({expected_h}, {expected_w})"
            )
        
        # Spatial upsampling: [T, C, h, w] -> [T, C, H, W].
        lq_spatial = F.interpolate(
            lq_video,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )

        # Temporal upsampling: [T, C, H, W] -> [1, C, T, H, W] -> target T.
        lq_bcthw = lq_spatial.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        if t == self.num_frames:    # time upsampling = 1，直接返回
            lq_up = lq_bcthw
        else:
            lq_up = temporal_blending(lq_bcthw, self.temporal_scale)
        return lq_up.squeeze(0).permute(1, 0, 2, 3).contiguous()

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to a video.

        Args:
            frames (torch.Tensor): A 4D tensor representing a video
                with shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed video tensor with the same shape as the input
        """
        return torch.stack([self.__frame_transform(f) for f in frames], dim=0)
    

    def init_degradation(self, opt) -> None:
        # Initialize first degradation operations
        self.random_blur_1 = RandomBlur(
            params=opt['degradation_1']['random_blur']['params'],
            keys=opt['degradation_1']['random_blur']['keys']
        )
        self.random_resize_1 = RandomResize(
            params=opt['degradation_1']['random_resize']['params'],
            keys=opt['degradation_1']['random_resize']['keys']
        )
        self.random_noise_1 = RandomNoise(
            params=opt['degradation_1']['random_noise']['params'],
            keys=opt['degradation_1']['random_noise']['keys']
        )
        self.random_jpeg_1 = RandomJPEGCompression(
            params=opt['degradation_1']['random_jpeg']['params'],
            keys=opt['degradation_1']['random_jpeg']['keys']
        )
        self.random_mpeg_1 = RandomVideoCompression(
            params=opt['degradation_1']['random_mpeg']['params'],
            keys=opt['degradation_1']['random_mpeg']['keys']
        )
        
        # Initialize second degradation operations
        self.random_blur_2 = RandomBlur(
            params=opt['degradation_2']['random_blur']['params'],
            keys=opt['degradation_2']['random_blur']['keys']
        )
        self.random_resize_2 = RandomResize(
            params=opt['degradation_2']['random_resize']['params'],
            keys=opt['degradation_2']['random_resize']['keys']
        )
        self.random_noise_2 = RandomNoise(
            params=opt['degradation_2']['random_noise']['params'],
            keys=opt['degradation_2']['random_noise']['keys']
        )
        self.random_jpeg_2 = RandomJPEGCompression(
            params=opt['degradation_2']['random_jpeg']['params'],
            keys=opt['degradation_2']['random_jpeg']['keys']
        )
        self.degradation_with_shuffle = DegradationsWithShuffle(
            degradations=opt['degradation_2']['degradation_with_shuffle']['degradations'],
            keys=opt['degradation_2']['degradation_with_shuffle']['keys']
        )
        
        # Define degradation sequence
        self.first_stage = [
            self.random_blur_1,
            self.random_resize_1, 
            self.random_noise_1,
            self.random_jpeg_1,
            self.random_mpeg_1
        ]
        
        self.second_stage = [
            self.random_blur_2,
            self.random_resize_2,
            self.random_noise_2,
            self.random_jpeg_2,
            self.degradation_with_shuffle
        ]

    def degrade(self, data):
        """
        Apply degradation pipeline to input data
        
        Args:
            data: dict containing frames to be processed (e.g., {'lqs': frame_list})
        
        Returns:
            dict: processed data with degraded frames
        """
        # Apply first stage degradations
        for degradation in self.first_stage:
            data = degradation(data)
        
        # Apply second stage degradations
        for degradation in self.second_stage:
            data = degradation(data)
            
        return data

    @staticmethod
    def to_tensor(frame: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame).float()
        else:
            frame = frame.float()
        if frame.ndim == 3 and frame.shape[-1] == 3:  # [H, W, C]
            frame = frame.permute(2, 0, 1).contiguous()  # [C, H, W]
        frame = torch.clamp(frame, 0.0, 255.0)
        return frame

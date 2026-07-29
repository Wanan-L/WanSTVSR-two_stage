import math
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
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
from .utils import (
    paired_random_crop_video,
    random_crop_frames,
    read_video_frames,
    scan_video_or_frame_dirs,
    scan_images,
    read_video_or_image,
    temporal_blending
)


class SpatioTemporalRealSRImageVideoDataset(Dataset):
    """Image-video mixed dataset with spatial degradation and video temporal degradation.

    The implementation follows the difference between `RealSRDataset` and
    `RealSRImageVideoDataset`, while keeping the data organization and relative
    imports of `SpatioTemporalRealSRDataset`:
      1. video samples use the two-stage RealBasicVSR spatial degradation;
      2. video samples are temporally downsampled and then temporally blended;
      3. image samples skip video compression and temporal downsampling;
      4. image samples use the additional resize-and-blur third stage;
      5. image and video lists are repeated to the same length and returned together.

    Returned HQ/LQ tensors are in [C, T, H, W] and normalized to [-1, 1].
    `lq_video` is restored to the HQ spatial size and frame count. `lq_image`
    is restored only spatially and keeps one frame.
    """

    def __init__(
        self,
        data_root: str | Path | None = None,
        image_data_root: str | Path | None = None,
        num_frames: int = 17,
        height: int = 320,
        width: int = 640,
        spatial_scale: int = 4,
        temporal_scale: int = 2,
        degradation_config: str | Path | None = None,
        inter_frame_margin: int = 10,
        image_exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", '.tiff', ".webp"),
    ) -> None:
        super().__init__()
        if data_root is None:
            raise ValueError("data_root must be provided")
        if image_data_root is None:
            raise ValueError("image_data_root must be provided")
        if degradation_config is None:
            raise ValueError("degradation_config must be provided")

        self.data_root = Path(data_root)
        self.image_data_root = Path(image_data_root)
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

        self.image_exts = tuple(ext.lower() for ext in image_exts)
        self.videos = scan_video_or_frame_dirs(self.data_root)
        self.images = scan_images(self.image_data_root)

        if not self.videos:
            raise FileNotFoundError(f"No video/frame samples found in {self.data_root}")
        if not self.images:
            raise FileNotFoundError(f"No image samples found in {self.image_data_root}")

        # Keep image and video samples one-to-one, following RealSRImageVideoDataset.
        if len(self.images) > len(self.videos):
            repeat_times = math.ceil(len(self.images) / len(self.videos))
            self.videos = (self.videos * repeat_times)[: len(self.images)]
        if len(self.videos) > len(self.images):
            repeat_times = math.ceil(len(self.videos) / len(self.images))
            self.images = (self.images * repeat_times)[: len(self.videos)]

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
        video_path = self.videos[index]
        image_path = self.images[index]

        # [C, T, H, W]
        image_lq, image_hq = self.preprocess_image_video(image_path, mode="image")
        video_lq, video_hq = self.preprocess_image_video(video_path, mode="video")

        return {
            "hq_video": video_hq,
            "lq_video": video_lq,
            "hq_image": image_hq,
            "lq_image": image_lq,
            "video_path": str(video_path),
            "image_path": str(image_path),
            "video_metadata": {
                "num_frames": self.num_frames,
                "height": self.height,
                "width": self.width,
            },
        }

    def preprocess_image_video(self, item_path: Path, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess one image or video and return normalized LQ/HQ [C, T, H, W]."""
        item_hq_frames, item_lq_frames = self.preprocess(item_path, mode)
        item_lq_frames_resize = self.upsample_lq_video(item_lq_frames, mode)

        item_hq = self.video_transform(item_hq_frames).permute(1, 0, 2, 3).contiguous()
        item_lq = self.video_transform(item_lq_frames_resize).permute(1, 0, 2, 3).contiguous()
        return item_lq, item_hq

    def preprocess(self, video_path: Path, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create paired HQ/LQ tensors in [T, C, H, W]."""
        if mode == "image":
            frame_list = read_video_or_image(video_path)
            sample_frames = 1
        elif mode == "video":
            frame_list = read_video_frames(video_path, self.inter_frames, self.image_exts)
            sample_frames = self.num_frames
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        crop_frame_list = random_crop_frames(
            frame_list,
            1 if mode == "image" else self.inter_frames,
            self.inter_height,
            self.inter_width,
        )
        inter_h, inter_w, _ = crop_frame_list[0].shape
        inter_target_h = inter_h // self.spatial_scale
        inter_target_w = inter_w // self.spatial_scale

        if mode == "image":
            self.random_resize_3.params['target_size'] = (inter_target_h, inter_target_w)
        else:
            if isinstance(self.degradation_with_shuffle.degradations[0],list):
                self.degradation_with_shuffle.degradations[0][0].params['target_size'] = (inter_target_h, inter_target_w)
            else:
                self.degradation_with_shuffle.degradations[1][0].params['target_size'] = (inter_target_h, inter_target_w)

        input_dict = dict(lqs=crop_frame_list)
        deg_frame_list = self.degrade(input_dict, mode)["lqs"]

        hq_frame_list, lq_frame_list = paired_random_crop_video(
            crop_frame_list,
            deg_frame_list,
            sample_frames,
            self.target_h,
            self.target_w,
            self.spatial_scale,
            str(video_path),
        )

        hq_tensor = torch.stack([self.to_tensor(frame) for frame in hq_frame_list], dim=0)
        lq_tensor = torch.stack([self.to_tensor(frame) for frame in lq_frame_list], dim=0)

        if hq_tensor.ndim != 4 or lq_tensor.ndim != 4:
            raise ValueError(
                "Expected HQ/LQ tensors in [T, C, H, W], got "
                f"HQ={tuple(hq_tensor.shape)}, LQ={tuple(lq_tensor.shape)}"
            )
        if hq_tensor.shape[0] != sample_frames or lq_tensor.shape[0] != sample_frames:
            raise ValueError(
                f"Expected {sample_frames} paired frames for mode={mode}, got "
                f"HQ={hq_tensor.shape[0]}, LQ={lq_tensor.shape[0]}"
            )

        if mode == "video":
            lq_tensor = lq_tensor[:: self.temporal_scale].contiguous()
            expected_frames = (self.num_frames - 1) // self.temporal_scale + 1
            if lq_tensor.shape[0] != expected_frames:
                raise RuntimeError(f"Expected {expected_frames} low-fps frames, got {lq_tensor.shape[0]}")

        return hq_tensor, lq_tensor

    def upsample_lq_video(self, lq_video: torch.Tensor, mode: str) -> torch.Tensor:
        """Restore raw LQ spatially; restore frame count only for video samples."""
        t, _, h, w = lq_video.shape
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
        if mode == "image" or t == self.num_frames:
            lq_up = lq_bcthw
        elif mode == "video":
            lq_up = temporal_blending(lq_bcthw, self.temporal_scale)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        lq_up = lq_up.squeeze(0).permute(1, 0, 2, 3).contiguous()
        expected_t = 1 if mode == "image" else self.num_frames
        if lq_up.shape[0] != expected_t:
            raise RuntimeError(
                f"Expected {expected_t} restored frames for mode={mode}, got {lq_up.shape[0]}"
            )
        return lq_up

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transform(frame) for frame in frames], dim=0)

    def init_degradation(self, opt) -> None:
        # Initialize first degradation operations.
        self.random_blur_1 = RandomBlur(
            params=opt["degradation_1"]["random_blur"]["params"],
            keys=opt["degradation_1"]["random_blur"]["keys"],
        )
        self.random_resize_1 = RandomResize(
            params=opt["degradation_1"]["random_resize"]["params"],
            keys=opt["degradation_1"]["random_resize"]["keys"],
        )
        self.random_noise_1 = RandomNoise(
            params=opt["degradation_1"]["random_noise"]["params"],
            keys=opt["degradation_1"]["random_noise"]["keys"],
        )
        self.random_jpeg_1 = RandomJPEGCompression(
            params=opt["degradation_1"]["random_jpeg"]["params"],
            keys=opt["degradation_1"]["random_jpeg"]["keys"],
        )
        # Video compression is not applied to image samples.
        self.random_mpeg_1 = RandomVideoCompression(
            params=opt["degradation_1"]["random_mpeg"]["params"],
            keys=opt["degradation_1"]["random_mpeg"]["keys"],
        )

        # Initialize second degradation operations.
        self.random_blur_2 = RandomBlur(
            params=opt["degradation_2"]["random_blur"]["params"],
            keys=opt["degradation_2"]["random_blur"]["keys"],
        )
        self.random_resize_2 = RandomResize(
            params=opt["degradation_2"]["random_resize"]["params"],
            keys=opt["degradation_2"]["random_resize"]["keys"],
        )
        self.random_noise_2 = RandomNoise(
            params=opt["degradation_2"]["random_noise"]["params"],
            keys=opt["degradation_2"]["random_noise"]["keys"],
        )
        self.random_jpeg_2 = RandomJPEGCompression(
            params=opt["degradation_2"]["random_jpeg"]["params"],
            keys=opt["degradation_2"]["random_jpeg"]["keys"],
        )
        self.degradation_with_shuffle = DegradationsWithShuffle(
            degradations=opt["degradation_2"]["degradation_with_shuffle"]["degradations"],
            keys=opt["degradation_2"]["degradation_with_shuffle"]["keys"],
        )

        # Initialize the image-only third degradation stage.
        self.random_resize_3 = RandomResize(
            params=opt["degradation_3"]["random_resize"]["params"],
            keys=opt["degradation_3"]["random_resize"]["keys"],
        )
        self.random_blur_3 = RandomBlur(
            params=opt["degradation_3"]["random_blur"]["params"],
            keys=opt["degradation_3"]["random_blur"]["keys"],
        )

        # Define degradation sequence
        self.first_stage = [
            self.random_blur_1,
            self.random_resize_1,
            self.random_noise_1,
            self.random_jpeg_1,
        ]
        self.second_stage = [
            self.random_blur_2,
            self.random_resize_2,
            self.random_noise_2,
            self.random_jpeg_2,
        ]
        self.third_shuffle = [self.degradation_with_shuffle]
        self.third_stage = [self.random_resize_3, self.random_blur_3]

    def degrade(self, data, mode: str):
        # Apply first stage degradations
        for degradation in self.first_stage:
            data = degradation(data)

        if mode == "video":
            data = self.random_mpeg_1(data)

        # Apply second stage degradations
        for degradation in self.second_stage:
            data = degradation(data)

        # Apply third stage degradations
        if mode == "video":
            for degradation in self.third_shuffle:
                data = degradation(data)
        elif mode == "image":
            for degradation in self.third_stage:
                data = degradation(data)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

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

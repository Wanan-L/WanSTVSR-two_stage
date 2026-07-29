import logging
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2, os, tqdm
import numpy as np
import torch  # noqa: F401
from PIL import Image
import decord
import imageio
from einops import rearrange
import torch.nn.functional as F
import imageio.v3 as iio

decord.bridge.set_bridge("torch")

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", '.tiff', ".webp")

def is_video(path): 
    return os.path.isfile(path) and path.lower().endswith(VIDEO_EXTS)

def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device='cuda'):
    """Convert a PIL image to a torch tensor in the range [-1, 1]"""    # 用来读取视频转成 tensor
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)
    t = t.permute(2,0,1) / 255.0 * 2.0 - 1.0
    return t.to(dtype)

def tensor2video(frames: torch.Tensor):
    """Convert a tensor of frames to a list of PIL images"""    # 用来保存视频
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames]
    return frames

def bcthw_to_fchw_01(video: torch.Tensor) -> torch.Tensor:
    """Convert [B, C, T, H, W], range [-1, 1], to [T, C, H, W], range [0, 1].""" # 用来计算指标
    if video.dim() != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {tuple(video.shape)}")
    video = video[0].float().clamp(-1, 1)
    video = ((video + 1.0) * 0.5).clamp(0, 1)
    return video.permute(1, 0, 2, 3).contiguous()


def scan_video_or_frame_dirs(
    root: str | Path,
    video_exts: Sequence[str] = VIDEO_EXTS,
    image_exts: Sequence[str] = IMAGE_EXTS,
) -> List[Path]:
    """Scan video or frame directories. Returns a list of paths."""
    root = Path(root)
    if not root.exists():
        return []

    samples: List[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in video_exts:
            samples.append(path)
        elif path.is_dir() and any(
            p.is_file() and p.suffix.lower() in image_exts
            for p in path.iterdir()
        ):
            samples.append(path)

    return samples

def scan_images(data_root: str | Path, image_exts: Sequence[str] = IMAGE_EXTS) -> List[Path]:
    """Recursively scan standalone image files in deterministic order."""
    data_root = Path(data_root)
    if not data_root.exists():
        return []

    return sorted(
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in image_exts
    )


def read_video_as_tensor(video_path: str | Path, dtype: torch.dtype = torch.bfloat16, device: str | torch.device = "cuda"):
    """Read a video as [1, C, T, H, W], range [-1, 1]."""
    video_path = str(video_path)

    if not is_video(video_path):
        raise ValueError(f"Unsupported video path: {video_path}")

    reader = imageio.get_reader(video_path)     # [H, W, C]
    try:
        meta = reader.get_meta_data()
    except Exception:
        meta = {}

    fps_val = meta.get("fps", 30)   # 获取 fps 
    fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

    frames = []
    try:
        for frame in reader:
            img = Image.fromarray(frame).convert("RGB")
            frames.append(pil_to_tensor_neg1_1(img, dtype=dtype, device=device))    # [H, W, C], 0~255 -> [C, H, W], [-1, 1]
    finally:
        try:
            reader.close()
        except Exception:
            pass

    if not frames:
        raise RuntimeError(f"No frames read from video: {video_path}")

    video = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0).contiguous()    # [1, C, T, H, W]

    return video, fps


def temporal_blending(video: torch.Tensor, temporal_scale: int) -> torch.Tensor:
      """Temporal linear blending for [B, C, T, H, W]."""
      if video.ndim != 5:
          raise ValueError(f"Expected [B,C,T,H,W], got {tuple(video.shape)}")
      b, c, t, h, w = video.shape
      if c != 3:
          raise ValueError(f"Expected RGB video with C=3, got C={c}")
      if temporal_scale <= 1 or t <= 1:
          return video

      frames = video.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
      weights = [i / temporal_scale for i in range(1, temporal_scale)]  # 插值帧索引，weights = [1/temporal_scale, 2/temporal_scale, ..., (temporal_scale-1)/temporal_scale]

      seq_outputs = []
      for idx in range(t - 1):
          im0 = frames[:, idx]      # 
          im1 = frames[:, idx + 1]
          if idx == 0:
              seq_outputs.append(im0.unsqueeze(1))
          for alpha in weights:
              middle = (1.0 - alpha) * im0 + alpha * im1    # 计算插值帧
              seq_outputs.append(middle.unsqueeze(1))
          seq_outputs.append(im1.unsqueeze(1))

      return torch.cat(seq_outputs, dim=1).permute(0, 2, 1, 3, 4).contiguous()


def resize_video_bcthw(
    video: torch.Tensor,
    spatial_scale: int | None = None,
    temporal_scale: int | None = None,
    mode_spatial: str = "bicubic",
) -> torch.Tensor:
    """Resize [B,C,T,H,W] video in temporal and spatial dimensions.The input and output value range are unchanged, e.g. [-1,1]."""
    
    if video.dim() != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {tuple(video.shape)}")

    b, c, t, h, w = video.shape

    target_h, target_w = h * spatial_scale, w * spatial_scale
    target_t = (t - 1) * temporal_scale + 1

    # Spatial resize: [B,C,T,H,W] -> [B*T,C,H,W] -> [B,C,T,H',W']
    if (target_h, target_w) != (h, w):
        x = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = F.interpolate(
            x.float(),
            size=(target_h, target_w),
            mode=mode_spatial,
            align_corners=False if mode_spatial in ("bilinear", "bicubic") else None,
        )
        video = x.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4).contiguous()

    # Temporal resize: 
    if target_t != video.shape[2]:
        b, c, t, h, w = video.shape
        video = temporal_blending(video, temporal_scale)

    return video.contiguous()


def prepare_input_tensor(
    lq_path: str | Path,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    spatial_scale: int = 4,
    temporal_scale: int = 2,
    mode_spatial: str = "bicubic",
):
    """Read raw degraded LQ video and upsample it to WanSTVSR model input size."""
    lq_video, fps = read_video_as_tensor(lq_path, dtype=dtype, device=device)

    lq_video = resize_video_bcthw(lq_video, spatial_scale, temporal_scale, mode_spatial)

    return lq_video.to(dtype=dtype, device=device).clamp(-1, 1), fps


def read_video_frames(video_path, frame_size=None, image_exts=IMAGE_EXTS):
    """
    Read a video file or frame directory Read video and pad frames if necessary.
    Returns:
        list of np.ndarray: Each frame is [H, W, C], dtype=uint8"""
    video_path = Path(video_path)
    frames = []

    if video_path.is_dir():
        files = [p for p in sorted(video_path.iterdir()) if p.suffix.lower() in image_exts]
        for file in files:
            frame = cv2.imread(str(file), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read image frame: {file}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        try:
            video_reader = decord.VideoReader(video_path.as_posix())
            if len(video_reader) > 0:
                frames = video_reader.get_batch(list(range(len(video_reader)))).numpy()
                frames = [frame for frame in frames]
        except Exception as error:
            raise RuntimeError(f"Decord failed to read video: {video_path}") from error

    if len(frames) == 0:
        raise RuntimeError(f"No frames read from: {video_path}")
    
    if frame_size is not None and len(frames) < frame_size:
        logging.warning("Sample %s has %d frames, padding to %d", video_path, len(frames), frame_size)
        last_frame = frames[-1]
        frames.extend([last_frame.copy() for _ in range(frame_size - len(frames))])

    return frames


def read_video_or_image(input_path, frame_size=None, video_exts=VIDEO_EXTS, image_exts=IMAGE_EXTS):
    """
    Read frames from a video file or a single image.

    Args:
        input_path (str or Path): Path to a video or image file.
        frame_size (int, optional): Desired number of frames. If fewer, pad with the last frame.

    Returns:
        list of np.ndarray: Each frame is [H, W, C], dtype=uint8
    """
    if isinstance(input_path, str):
        input_path = Path(input_path)
    
    if input_path.suffix.lower() in video_exts:
        # Read from video
        video_reader = decord.VideoReader(str(input_path))
        frames = video_reader.get_batch(list(range(len(video_reader)))).numpy()
        frame_list = [frame for frame in frames]
    elif input_path.suffix.lower() in image_exts:
        # Read from single image
        img = Image.open(input_path).convert("RGB")
        frame_np = np.array(img, dtype=np.uint8)
        frame_list = [frame_np]
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}")
    
    # 帧数不足时使用最后一帧补齐
    if frame_size is not None and len(frame_list) < frame_size:
        logging.warning("Sample %s has %d frames, padding to %d", input_path, len(frame_list), frame_size)
        last_frame = frame_list[-1]
        frame_list.extend([last_frame.copy() for _ in range(frame_size - len(frames))])

    return frame_list  # [F, H, W, C] uint8


def random_crop_frames(
    frames: List[np.ndarray],
    frame_size: int,
    height: int,
    width: int,
) -> List[np.ndarray]:
    """Randomly crop a temporal clip and spatial window from RGB frames."""
    total = len(frames)
    h, w, _ = frames[0].shape
    if total < frame_size:
        raise ValueError(f"Expected at least {frame_size} frames, got {total}")

    start = random.randint(0, total - frame_size)
    crop_h = min(height, h)
    crop_w = min(width, w)
    crop_h -= crop_h % 4
    crop_w -= crop_w % 4
    top = random.randint(0, h - crop_h) if h > crop_h else 0
    left = random.randint(0, w - crop_w) if w > crop_w else 0

    return [frame[top : top + crop_h, left : left + crop_w, :] for frame in frames[start : start + frame_size]]


def paired_random_crop_video(
    hq_frames: List[np.ndarray],
    lq_frames: List[np.ndarray],
    num_frames: int,
    lq_crop_h: int,
    lq_crop_w: int,
    scale: int,
    file_path: str | None = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Paired random crop for aligned HQ/LQ video frames."""
    assert len(hq_frames) == len(lq_frames), "HQ and LQ must have same number of frames"
    assert len(hq_frames) >= num_frames, "Not enough frames for temporal crop"

    h_lq, w_lq, _ = lq_frames[0].shape
    h_hq, w_hq, _ = hq_frames[0].shape
    assert h_hq == h_lq * scale and w_hq == w_lq * scale, (
        f"File [{file_path}]: Spatial size mismatch: HQ ({h_hq}, {w_hq}) vs "
        f"LQ ({h_lq}, {w_lq}) with scale {scale}"
    )
    assert h_lq >= lq_crop_h and w_lq >= lq_crop_w, f"File [{file_path}]: LQ crop size too large"

    top = random.randint(0, h_lq - lq_crop_h)
    left = random.randint(0, w_lq - lq_crop_w)
    top_hq, left_hq = top * scale, left * scale
    hq_crop_h, hq_crop_w = lq_crop_h * scale, lq_crop_w * scale
    start = random.randint(0, len(hq_frames) - num_frames)

    cropped_hq = [
        frame[top_hq : top_hq + hq_crop_h, left_hq : left_hq + hq_crop_w, :]
        for frame in hq_frames[start : start + num_frames]
    ]
    cropped_lq = [
        frame[top : top + lq_crop_h, left : left + lq_crop_w, :]
        for frame in lq_frames[start : start + num_frames]
    ]

    return cropped_hq, cropped_lq


def save_video(video: torch.Tensor, path: str, fps: float):
    if video.dim() == 5:
        video = video[0]  # [C, T, H, W]
        
    video = video.detach().float().cpu().clamp(-1, 1)
    video = (video + 1.0) * 0.5          # [-1, 1] -> [0, 1]
    video = video.permute(1, 2, 3, 0)    # [C, T, H, W] -> [T, H, W, C]
    video = (video * 255.0).round().to(torch.uint8).numpy()

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    iio.imwrite(
        path,
        video,
        fps=float(fps),
        codec="libx264",
        macro_block_size=1,
        ffmpeg_params=[
            "-crf", "0",
        ]
    )

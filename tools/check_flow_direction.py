#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import sys

sys.path.append("/data2/wujialing/project/STVSR/WanSTVSR")

from utils.RAFT.raft_bi import RAFT_bi
from utils.optical_flow_utils import flow_warp


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def read_video_or_frames(path: str, num_frames: int = 0) -> List[np.ndarray]:
    path = Path(path)

    frames: List[np.ndarray] = []

    if path.is_dir():
        files = [p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        if not files:
            raise FileNotFoundError(f"No image frames found in {path}")

        for file in files:
            img = Image.open(file).convert("RGB")
            frames.append(np.array(img))

    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from {path}")

    if num_frames > 0:
        if len(frames) >= num_frames:
            frames = frames[:num_frames]
        else:
            last = frames[-1]
            frames = frames + [last.copy() for _ in range(num_frames - len(frames))]

    return frames


def frames_to_tensor(
    frames: List[np.ndarray],
    height: int = 0,
    width: int = 0,
    device: str = "cuda",
) -> torch.Tensor:
    tensors = []

    for frame in frames:
        if height > 0 and width > 0:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        arr = torch.from_numpy(frame).float() / 255.0
        arr = arr.permute(2, 0, 1).contiguous()
        tensors.append(arr)

    video = torch.stack(tensors, dim=1).unsqueeze(0)
    # [1, C, T, H, W], range [0, 1]
    return video.to(device)


def unpack_raft_output(output: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, dict):
        flows_forward = (
            output.get("flows_forward")
            or output.get("forward")
            or output.get("flow_forward")
        )
        flows_backward = (
            output.get("flows_backward")
            or output.get("backward")
            or output.get("flow_backward")
        )
    elif isinstance(output, (tuple, list)) and len(output) >= 2:
        flows_forward, flows_backward = output[0], output[1]

        if isinstance(flows_forward, dict):
            output = flows_forward
            flows_forward = (
                output.get("flows_forward")
                or output.get("forward")
                or output.get("flow_forward")
            )
            flows_backward = (
                output.get("flows_backward")
                or output.get("backward")
                or output.get("flow_backward")
            )
    else:
        raise RuntimeError(f"Unsupported RAFT output type: {type(output)}")

    if flows_forward is None or flows_backward is None:
        raise RuntimeError("RAFT_bi output does not contain forward/backward flows.")

    return flows_forward, flows_backward


def resize_flow_to_video(flow: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    flow: [B, 2, T, Hf, Wf]
    return: [B, 2, T, height, width]
    """
    if flow.dim() != 5:
        raise ValueError(f"Expected flow [B,2,T,H,W], got {tuple(flow.shape)}")

    b, c, t, h0, w0 = flow.shape

    flow_4d = flow.permute(0, 2, 1, 3, 4).reshape(b * t, c, h0, w0)
    flow_4d = F.interpolate(
        flow_4d,
        size=(height, width),
        mode="bilinear",
        align_corners=True,
    )

    flow_4d[:, 0] *= float(width) / max(float(w0), 1.0)
    flow_4d[:, 1] *= float(height) / max(float(h0), 1.0)

    return flow_4d.reshape(b, t, c, height, width).permute(0, 2, 1, 3, 4).contiguous()


def to_uint8_image(x: torch.Tensor) -> np.ndarray:
    """
    x: [C, H, W], range [0, 1]
    """
    x = x.detach().float().clamp(0, 1).cpu()
    x = (x * 255.0).round().to(torch.uint8)
    return x.permute(1, 2, 0).numpy()


def save_debug_images(
    save_dir: Path,
    video: torch.Tensor,
    flows_forward: torch.Tensor,
    flows_backward: torch.Tensor,
    pair_idx: int,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    frame_i = video[:, :, pair_idx]
    frame_next = video[:, :, pair_idx + 1]

    flow_f = flows_forward[:, :, pair_idx]
    flow_b = flows_backward[:, :, pair_idx]

    warp_next_normal = flow_warp(frame_next, flow_f.permute(0, 2, 3, 1))
    warp_next_swapped = flow_warp(frame_next, flow_b.permute(0, 2, 3, 1))

    diff_normal = (warp_next_normal - frame_i).abs()
    diff_swapped = (warp_next_swapped - frame_i).abs()

    Image.fromarray(to_uint8_image(frame_i[0])).save(save_dir / "frame_i.png")
    Image.fromarray(to_uint8_image(frame_next[0])).save(save_dir / "frame_i_plus_1.png")

    Image.fromarray(to_uint8_image(warp_next_normal[0])).save(save_dir / "warp_next_normal.png")
    Image.fromarray(to_uint8_image(diff_normal[0])).save(save_dir / "diff_normal.png")

    Image.fromarray(to_uint8_image(warp_next_swapped[0])).save(save_dir / "warp_next_swapped.png")
    Image.fromarray(to_uint8_image(diff_swapped[0])).save(save_dir / "diff_swapped.png")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="HQ/GT video path or frame directory.")
    parser.add_argument("--raft_ckpt_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--vis_pair_idx", type=int, default=0)
    args = parser.parse_args()

    device = args.device
    frames = read_video_or_frames(args.video, num_frames=args.num_frames)
    video = frames_to_tensor(frames, height=args.height, width=args.width, device=device)

    b, c, t, h, w = video.shape
    if t < 2:
        raise ValueError("Need at least 2 frames to check flow direction.")

    raft = RAFT_bi(args.raft_ckpt_path, device=device)
    if hasattr(raft, "eval"):
        raft.eval()
    if hasattr(raft, "requires_grad_"):
        raft.requires_grad_(False)

    output = raft(video)
    flows_forward, flows_backward = unpack_raft_output(output)

    flows_forward = resize_flow_to_video(flows_forward.to(device), h, w)
    flows_backward = resize_flow_to_video(flows_backward.to(device), h, w)

    normal_errors = []
    swapped_errors = []

    for i in range(t - 1):
        frame_i = video[:, :, i]
        frame_next = video[:, :, i + 1]

        flow_f = flows_forward[:, :, i]
        flow_b = flows_backward[:, :, i]

        # This is the same direction usage as your current training code.
        warp_next_normal = flow_warp(frame_next, flow_f.permute(0, 2, 3, 1))
        warp_prev_normal = flow_warp(frame_i, flow_b.permute(0, 2, 3, 1))

        err_normal = (
            (warp_next_normal - frame_i).abs().mean()
            + (warp_prev_normal - frame_next).abs().mean()
        )

        # Swapped usage.
        warp_next_swapped = flow_warp(frame_next, flow_b.permute(0, 2, 3, 1))
        warp_prev_swapped = flow_warp(frame_i, flow_f.permute(0, 2, 3, 1))

        err_swapped = (
            (warp_next_swapped - frame_i).abs().mean()
            + (warp_prev_swapped - frame_next).abs().mean()
        )

        normal_errors.append(err_normal)
        swapped_errors.append(err_swapped)

        print(
            f"pair {i:02d}-{i+1:02d}: "
            f"normal={err_normal.item():.6f}, "
            f"swapped={err_swapped.item():.6f}"
        )

    normal = torch.stack(normal_errors).mean().item()
    swapped = torch.stack(swapped_errors).mean().item()

    print("\n=== Flow direction check ===")
    print(f"normal mean error : {normal:.6f}")
    print(f"swapped mean error: {swapped:.6f}")

    if normal <= swapped:
        print("\nResult: current training direction is likely correct.")
        print("Keep --swap_flow_directions disabled.")
    else:
        print("\nResult: flow direction is likely swapped.")
        print("Enable --swap_flow_directions in training.")

    if args.save_dir:
        pair_idx = min(max(args.vis_pair_idx, 0), t - 2)
        save_debug_images(
            Path(args.save_dir),
            video,
            flows_forward,
            flows_backward,
            pair_idx,
        )
        print(f"\nSaved debug images to: {args.save_dir}")


if __name__ == "__main__":
    main()
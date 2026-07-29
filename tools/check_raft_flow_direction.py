#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone diagnostic for WanSTVSR flow direction.

Run this before TE training. It performs:
1. A deterministic one-pixel flow_warp sign test.
2. An optional RAFT_bi output-direction diagnostic on a translated textured pair.

The script does not modify training code or model weights.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Tuple

sys.path.append("/data2/wujialing/project/STVSR/WanSTVSR-0713")

import torch
import torch.nn.functional as F

from utils.RAFT.raft_bi import RAFT_bi
from utils.optical_flow_utils import flow_warp


def normalize_flow_shape(flow: torch.Tensor) -> torch.Tensor:
    """Return flow as [B, 2, T, H, W]."""
    if flow.dim() != 5:
        raise ValueError(f"Expected a 5D flow tensor, got {tuple(flow.shape)}.")
    if flow.shape[1] == 2:
        return flow
    if flow.shape[2] == 2:
        return flow.permute(0, 2, 1, 3, 4).contiguous()
    raise ValueError(f"Cannot find a two-channel dimension in {tuple(flow.shape)}.")


def resize_flow(flow: torch.Tensor, height: int, width: int) -> torch.Tensor:
    flow = normalize_flow_shape(flow)[:, :, :1]
    batch, channels, pairs, old_height, old_width = flow.shape
    flat = (
        flow.permute(0, 2, 1, 3, 4)
        .reshape(batch * pairs, channels, old_height, old_width)
    )
    flat = F.interpolate(
        flat,
        size=(height, width),
        mode="bilinear",
        align_corners=True,
    )
    flat[:, 0] *= float(width) / max(float(old_width), 1.0)
    flat[:, 1] *= float(height) / max(float(old_height), 1.0)
    return flat.reshape(batch, pairs, channels, height, width)[:, 0]


def in_bounds_mask(flow: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = flow.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    sample_x = grid_x.unsqueeze(0) + flow[:, 0]
    sample_y = grid_y.unsqueeze(0) + flow[:, 1]
    return (
        (sample_x >= 0.0)
        & (sample_x <= width - 1)
        & (sample_y >= 0.0)
        & (sample_y <= height - 1)
    ).unsqueeze(1)


def masked_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    channels = prediction.shape[1]
    denominator = mask.sum() * channels + 1e-6
    return float(((prediction - target).abs() * mask).sum() / denominator)


def check_flow_warp_one_pixel(device: torch.device) -> None:
    source = torch.zeros(1, 1, 5, 7, device=device)
    target = torch.zeros_like(source)
    source[:, :, 2, 2] = 1.0
    target[:, :, 2, 3] = 1.0

    minus_one = torch.zeros(1, 5, 7, 2, device=device)
    minus_one[..., 0] = -1.0
    plus_one = torch.zeros_like(minus_one)
    plus_one[..., 0] = 1.0

    error_minus = (flow_warp(source, minus_one) - target).abs().max().item()
    error_plus = (flow_warp(source, plus_one) - target).abs().max().item()

    print("[flow_warp one-pixel test]")
    print(f"  horizontal flow -1 max error: {error_minus:.8f}")
    print(f"  horizontal flow +1 max error: {error_plus:.8f}")
    if error_minus > 1e-6 or error_plus <= 1e-6:
        raise RuntimeError("The expected flow_warp sign convention was not observed.")
    print("  PASS: flow_warp uses backward sampling: output(x)=source(x+flow(x)).")


def make_translated_pair(
    height: int,
    width: int,
    shift_x: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(1234)
    texture = torch.rand(1, 3, height, width, generator=generator, device=device)
    texture = F.avg_pool2d(texture, kernel_size=5, stride=1, padding=2)

    frame_0 = texture
    frame_1 = torch.zeros_like(frame_0)
    if shift_x > 0:
        frame_1[..., shift_x:] = frame_0[..., : width - shift_x]
    elif shift_x < 0:
        shift = -shift_x
        frame_1[..., : width - shift] = frame_0[..., shift:]
    else:
        frame_1.copy_(frame_0)
    return frame_0, frame_1


def unpack_raft_output(output: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, dict):
        forward = None
        backward = None
        for key in ("flows_forward", "forward", "flow_forward"):
            if output.get(key) is not None:
                forward = output[key]
                break
        for key in ("flows_backward", "backward", "flow_backward"):
            if output.get(key) is not None:
                backward = output[key]
                break
    elif isinstance(output, (list, tuple)) and len(output) == 2:
        forward, backward = output
    else:
        raise RuntimeError(
            f"Unsupported RAFT_bi output type: {type(output)}."
        )
    if forward is None or backward is None:
        raise RuntimeError("RAFT_bi did not return both directions.")
    return forward, backward


def evaluate_candidate(
    source: torch.Tensor,
    target: torch.Tensor,
    flow: torch.Tensor,
) -> float:
    mask = in_bounds_mask(flow).to(source.dtype)
    warped = flow_warp(source, flow.permute(0, 2, 3, 1))
    return masked_mae(warped, target, mask)


def check_raft_outputs(args: argparse.Namespace, device: torch.device) -> None:
    frame_0, frame_1 = make_translated_pair(
        args.height,
        args.width,
        args.shift_x,
        device,
    )
    video = torch.stack([frame_0, frame_1], dim=2)

    raft = RAFT_bi(args.raft_ckpt_path, device=device)
    raft.requires_grad_(False)
    raft.eval().float()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        output = raft(video.float())
    flow_forward, flow_backward = unpack_raft_output(output)
    flow_forward = resize_flow(flow_forward.float(), args.height, args.width)
    flow_backward = resize_flow(flow_backward.float(), args.height, args.width)

    no_warp_error = float((frame_0 - frame_1).abs().mean())
    errors: Dict[str, float] = {
        "forward: frame0 -> frame1": evaluate_candidate(
            frame_0, frame_1, flow_forward
        ),
        "backward: frame0 -> frame1": evaluate_candidate(
            frame_0, frame_1, flow_backward
        ),
        "forward: frame1 -> frame0": evaluate_candidate(
            frame_1, frame_0, flow_forward
        ),
        "backward: frame1 -> frame0": evaluate_candidate(
            frame_1, frame_0, flow_backward
        ),
    }

    print("\n[RAFT_bi output-direction diagnostic]")
    print(f"  synthetic horizontal translation: {args.shift_x:+d} pixels")
    print(f"  no-warp frame0/frame1 MAE: {no_warp_error:.8f}")
    for name, error in errors.items():
        print(f"  {name:<32} MAE: {error:.8f}")

    forward_to_next = errors["forward: frame0 -> frame1"]
    backward_to_next = errors["backward: frame0 -> frame1"]
    selected = "forward" if forward_to_next < backward_to_next else "backward"
    best = min(forward_to_next, backward_to_next)

    print("\n[recommended TE setting]")
    print(f'  FLOW_TO_NEXT_OUTPUT = "{selected}"')
    if best >= no_warp_error:
        print(
            "  WARNING: neither RAFT output improved the synthetic-pair warp. "
            "Repeat the diagnostic on a real adjacent-frame pair before fixing "
            "the training implementation."
        )
    else:
        print("  The selected output reduced the frame0 -> frame1 warp error.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check WanSTVSR flow direction.")
    parser.add_argument(
        "--raft_ckpt_path",
        type=str,
        default="./utils/RAFT/raft-things.pth",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--shift_x", type=int, default=8)
    parser.add_argument(
        "--skip_raft",
        action="store_true",
        help="Run only the deterministic flow_warp unit test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)
    check_flow_warp_one_pixel(device)
    if not args.skip_raft:
        check_raft_outputs(args, device)


if __name__ == "__main__":
    main()

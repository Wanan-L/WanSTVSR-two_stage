import json
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
import warnings
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings('ignore', category=FutureWarning, module='timm')

WANSTVSR_ROOT = Path(__file__).resolve().parents[1]
METRICS_ROOT = WANSTVSR_ROOT / 'metrics'
FR_METRICS = ['psnr', 'ssim', 'lpips', 'dists']
NR_METRICS = ['clipiqa', 'clipiqa+', 'niqe', 'ilniqe', 'liqe', 'musiq', 'maniqa', 'brisque']
TEMPORAL_METRICS = ['dover', 'ewarp', 'fastvqa', 'vbench', 'mdvqa']
ALL_METRICS = FR_METRICS + NR_METRICS + TEMPORAL_METRICS
VIDEO_EXTS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
IMAGE_EXTS = ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
TO_TENSOR = transforms.ToTensor()


def is_video_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def read_video_frames(video_path: str | Path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(TO_TENSOR(Image.fromarray(rgb)))
    cap.release()
    return torch.stack(frames) if frames else None


def read_image_folder(folder_path: str | Path):
    files = sorted([p for p in Path(folder_path).iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not files:
        return None
    frames = [TO_TENSOR(Image.open(p).convert('RGB')) for p in files]
    return torch.stack(frames)


def load_sequence(path: str | Path):
    path = Path(path)
    if path.is_dir():
        return read_image_folder(path)
    if path.is_file() and is_video_file(path):
        return read_video_frames(path)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return TO_TENSOR(Image.open(path).convert('RGB')).unsqueeze(0)
    return None


def rgb_to_y(img):
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return 0.257 * r + 0.504 * g + 0.098 * b + 0.0625


def crop_border(img, crop):
    if crop <= 0:
        return img
    if img.shape[-2] <= 2 * crop or img.shape[-1] <= 2 * crop:
        raise ValueError(f'Crop border {crop} is too large for input size {img.shape[-1]}x{img.shape[-2]}.')
    return img[:, :, crop:-crop, crop:-crop]


def crop_frames_center(frames, target_h, target_w):
    _, _, h, w = frames.shape
    top = max((h - target_h) // 2, 0)
    left = max((w - target_w) // 2, 0)
    return frames[:, :, top:top + target_h, left:left + target_w]


def crop_frames_top_left(frames, target_h, target_w):
    return frames[:, :, :target_h, :target_w]


def match_resolution(gt_frames, pred_frames, item_name='', is_center=False):
    t_gt_orig = gt_frames.shape[0]
    t_pred_orig = pred_frames.shape[0]
    t = min(t_gt_orig, t_pred_orig)
    gt_frames = gt_frames[:t]
    pred_frames = pred_frames[:t]
    _, _, h_g, w_g = gt_frames.shape
    _, _, h_p, w_p = pred_frames.shape
    target_h = min(h_g, h_p)
    target_w = min(w_g, w_p)

    crop_mode = 'center' if is_center else 'top-left'
    if item_name:
        print(
            f'[{item_name}] Pre-alignment - GT: {t_gt_orig} frames, {w_g}x{h_g} | '
            f'Input: {t_pred_orig} frames, {w_p}x{h_p} | crop mode: {crop_mode}'
        )

    if is_center:
        gt_frames = crop_frames_center(gt_frames, target_h, target_w)
        pred_frames = crop_frames_center(pred_frames, target_h, target_w)
    else:
        gt_frames = crop_frames_top_left(gt_frames, target_h, target_w)
        pred_frames = crop_frames_top_left(pred_frames, target_h, target_w)
    return gt_frames, pred_frames


def prepare_fr_eval_tensors(pred, gt, crop=0, test_y_channel=False):
    pred_eval = crop_border(pred, crop)
    gt_eval = crop_border(gt, crop)
    if test_y_channel:
        pred_eval = rgb_to_y(pred_eval)
        gt_eval = rgb_to_y(gt_eval)
    return pred_eval, gt_eval


def save_video(video_bcthw_01: torch.Tensor, path: str | Path, fps: int = 8, save_format: str = 'yuv444p') -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = video_bcthw_01[0].detach().cpu().clamp(0, 1).permute(1, 2, 3, 0)
    frames = frames.mul(255).round().byte().numpy()
    if path.suffix.lower() == '.mkv' or save_format == 'rgb_lossless':
        iio.imwrite(path, frames, fps=fps, codec='libx264rgb', pixelformat='rgb24', macro_block_size=None, ffmpeg_params=['-crf', '0'])
    else:
        iio.imwrite(path, frames, fps=fps, codec='libx264', pixelformat=save_format, macro_block_size=None, ffmpeg_params=['-crf', '0'])


def img2video(folder_path: str | Path, output_path: str | Path, fps: int = 25, save_format: str = 'yuv444p') -> bool:
    img_tensor = read_image_folder(folder_path)
    if img_tensor is None:
        return False
    video = img_tensor.permute(1, 0, 2, 3).unsqueeze(0)
    save_video(video, output_path, fps=fps, save_format=save_format)
    return True


def prepare_temp_videos(pred_dir: str | Path, output_dir: str | Path, fps: int = 25, save_format: str = 'yuv444p'):
    pred_dir = Path(pred_dir)
    temp_dir = Path(output_dir) / 'temp_for_video_metrics'
    temp_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in sorted(pred_dir.iterdir()):
        if item == temp_dir:
            continue
        key = item.stem
        dst = temp_dir / f'{key}.mp4'
        if dst.exists():
            paths.append(dst)
            continue
        if item.is_dir():
            if img2video(item, dst, fps=fps, save_format=save_format):
                paths.append(dst)
        elif item.is_file() and is_video_file(item):
            dst.symlink_to(item.resolve())
            paths.append(dst)
    return temp_dir, paths


def compute_pyiqa_metrics(
    pred_dir: str | Path,
    gt_dir: str | Path,
    metrics_list,
    device,
    batch_mode: bool = False,
    crop: int = 0,
    test_y_channel: bool = False,
    is_center: bool = False,
):
    import pyiqa

    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir) if gt_dir else None
    has_gt = gt_dir is not None and gt_dir.exists()
    needs_fr = any(metric in FR_METRICS for metric in metrics_list)
    if needs_fr and not has_gt:
        print(f'GT directory not found; skip full-reference metrics: {gt_dir}')
    iqa_models = {}
    for metric in metrics_list:
        if metric in FR_METRICS + NR_METRICS:
            try:
                iqa_models[metric] = pyiqa.create_metric(metric).to(device).eval()
            except Exception as exc:
                print(f'Failed to initialize {metric}: {exc}')
    results = {}
    items = [p for p in sorted(pred_dir.iterdir()) if p.is_dir() or is_video_file(p) or p.suffix.lower() in IMAGE_EXTS]
    matched_gt_count = 0
    for item in tqdm(items, desc='PyIQA metrics'):
        key = item.stem
        pred_seq = load_sequence(item)
        if pred_seq is None:
            continue
        gt_seq = None
        if has_gt:
            for cand in [gt_dir / item.name, gt_dir / key, gt_dir / f'{key}.mp4', gt_dir / f'{key}.mkv']:
                if cand.exists():
                    gt_seq = load_sequence(cand)
                    if gt_seq is not None:
                        break
        if gt_seq is not None:
            matched_gt_count += 1
            gt_seq, pred_seq = match_resolution(gt_seq, pred_seq, item_name=item.name, is_center=is_center)
            t_gt, _, h_gt, w_gt = gt_seq.shape
            t_pred, _, h_pred, w_pred = pred_seq.shape
            print(f'{key} - GT => frames: {t_gt}, resolution: {w_gt}x{h_gt}')
            print(f'{key} - Input => frames: {t_pred}, resolution: {w_pred}x{h_pred}')
            gt_seq = gt_seq.to(device)
        pred_seq = pred_seq.to(device)
        item_result = {}
        for name, model in iqa_models.items():
            try:
                if name in FR_METRICS:
                    if gt_seq is None:
                        continue
                    if batch_mode:
                        pred_eval, gt_eval = prepare_fr_eval_tensors(
                            pred_seq,
                            gt_seq,
                            crop=crop,
                            test_y_channel=test_y_channel,
                        )
                        values = model(pred_eval, gt_eval)
                        item_result[name] = float(values.mean().item() if torch.is_tensor(values) else values)
                    else:
                        frame_values = []
                        for idx in range(pred_seq.shape[0]):
                            pred_eval, gt_eval = prepare_fr_eval_tensors(
                                pred_seq[idx:idx + 1],
                                gt_seq[idx:idx + 1],
                                crop=crop,
                                test_y_channel=test_y_channel,
                            )
                            frame_values.append(model(pred_eval, gt_eval).item())
                        if frame_values:
                            item_result[name] = float(np.mean(frame_values))
                elif batch_mode:
                    values = model(pred_seq)
                    item_result[name] = float(values.mean().item() if torch.is_tensor(values) else values)
                else:
                    frame_values = []
                    for idx in range(pred_seq.shape[0]):
                        frame_values.append(model(pred_seq[idx:idx + 1]).item())
                    if frame_values:
                        item_result[name] = float(np.mean(frame_values))
            except Exception as exc:
                print(f'Error computing {name} for {key}: {exc}')
        results[key] = item_result
    if needs_fr and has_gt and matched_gt_count == 0:
        print(
            f'No GT samples matched predictions under {gt_dir}; '
            f'skip full-reference metrics {FR_METRICS}. '
            'Expected names like pred 000.mp4 matching GT 000.mp4, 000.mkv, or folder 000.'
        )
    return results


def compute_ewarp(video_paths, device):
    raft_root = METRICS_ROOT / 'RAFT'
    core_root = raft_root / 'core'
    sys.modules.pop("utils", None)

    results = {}
    if not (core_root / 'raft.py').exists():
        print(f'RAFT code not found at {core_root}; skip EWARP.')
        return results
    sys.path.insert(0, str(core_root))
    from raft import RAFT
    from utils.utils import InputPadder

    class Args:
        small = False
        mixed_precision = False
        alternate_corr = False
        dropout = 0

        def __contains__(self, key):
            return hasattr(self, key)

    model_path = raft_root / 'models' / 'raft-things.pth'
    if not model_path.exists():
        print(f'RAFT checkpoint not found at {model_path}; skip EWARP.')
        return results
    model = torch.nn.DataParallel(RAFT(Args()))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.module.to(device).eval()
    for video_path in tqdm(video_paths, desc='EWARP'):
        seq = load_sequence(video_path)
        if seq is None or seq.shape[0] < 2:
            continue
        seq = seq.to(device)
        errors = []
        for idx in range(seq.shape[0] - 1):
            image1 = seq[idx:idx + 1]
            image2 = seq[idx + 1:idx + 2]
            padder = InputPadder(image1.shape)
            image1_255, image2_255 = padder.pad(image1 * 255.0, image2 * 255.0)
            with torch.no_grad():
                _, flow = model(image1_255, image2_255, iters=20, test_mode=True)
            b, c, h, w = image1_255.shape
            grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
            grid = torch.stack((grid_x, grid_y), dim=-1).float().unsqueeze(0).repeat(b, 1, 1, 1)
            vgrid = grid + flow.permute(0, 2, 3, 1)
            vgrid[..., 0] = 2.0 * vgrid[..., 0] / max(w - 1, 1) - 1.0
            vgrid[..., 1] = 2.0 * vgrid[..., 1] / max(h - 1, 1) - 1.0
            image1_pad, image2_pad = padder.pad(image1, image2)
            warped = F.grid_sample(image2_pad, vgrid, align_corners=True)
            warped = padder.unpad(warped)
            errors.append(float(((image1 - warped) ** 2).mean().item()))
        results[video_path.stem] = {'ewarp': float(np.mean(errors)), 'ewarp_scaled_1000': float(np.mean(errors) * 1000.0)}
    return results


def compute_dover(video_paths, device):
    results = {}
    dover_dir = METRICS_ROOT / 'DOVER'
    if not dover_dir.exists():
        print(f'DOVER directory not found at {dover_dir}; skip DOVER.')
        return results
    sys.path.insert(0, str(dover_dir))
    try:
        import yaml
        from dover.datasets import ViewDecompositionDataset
        from dover.models import DOVER
    except Exception as exc:
        print(f'Failed to import DOVER: {exc}')
        return results
    yml_path = dover_dir / 'dover.yml'
    weight_path = dover_dir / 'pretrained_weights' / 'DOVER.pth'
    if not weight_path.exists():
        print(f'DOVER checkpoint not found at {weight_path}; skip DOVER.')
        return results
    with open(yml_path, 'r', encoding='utf-8') as f:
        opt = yaml.safe_load(f)
    model = DOVER(**opt['model']['args']).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    temp_list = Path(video_paths[0]).parent if video_paths else None
    if temp_list is None:
        return results
    dopt = opt['data']['val-l1080p']['args']
    dopt['anno_file'] = None
    dopt['data_prefix'] = str(temp_list)
    dopt['sample_types']['technical']['num_clips'] = 1
    dopt['sample_types']['technical']['frame_interval'] = 1
    dataset = ViewDecompositionDataset(dopt)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)
    for data in tqdm(loader, desc='DOVER'):
        if len(data.keys()) == 1:
            continue
        video = {}
        for key in ['aesthetic', 'technical']:
            if key in data:
                video[key] = data[key].to(device)
                b, c, t, h, w = video[key].shape
                video[key] = video[key].reshape(b, c, data['num_clips'][key], t // data['num_clips'][key], h, w).permute(0, 2, 1, 3, 4, 5).reshape(b * data['num_clips'][key], c, t // data['num_clips'][key], h, w)
        with torch.no_grad():
            res = model(video, reduce_scores=False)
            res = [np.mean(x.cpu().numpy()) for x in res]
        technical = (res[1] - 0.1107) / 0.07355
        aesthetic = (res[0] + 0.08285) / 0.03774
        overall = technical * 0.6104 + aesthetic * 0.3896
        key = Path(data['name'][0]).stem
        results[key] = {
            'dover_aesthetic': float(1 / (1 + np.exp(-aesthetic))),
            'dover_technical': float(1 / (1 + np.exp(-technical))),
            'dover': float(1 / (1 + np.exp(-overall))),
        }
    return results


def compute_fastvqa(video_paths, device):
    results = {}
    fastvqa_dir = METRICS_ROOT / 'FastVQA'
    if not fastvqa_dir.exists():
        print(f'FastVQA directory not found at {fastvqa_dir}; skip FastVQA.')
        return results
    sys.path.insert(0, str(fastvqa_dir))
    try:
        import decord
        import yaml
        from fastvqa.datasets import FragmentSampleFrames, SampleFrames, get_spatial_fragments
        from fastvqa.models import DiViDeAddEvaluator
    except Exception as exc:
        print(f'Failed to import FastVQA: {exc}')
        return results
    opt_path = fastvqa_dir / 'options' / 'fast' / 'f3dvqa-b.yml'
    weight_path = fastvqa_dir / 'pretrained_weights' / 'FAST_VQA_3D_1_1.pth'
    if not weight_path.exists():
        print(f'FastVQA checkpoint not found at {weight_path}; skip FastVQA.')
        return results
    with open(opt_path, 'r', encoding='utf-8') as f:
        opt = yaml.safe_load(f)
    model = DiViDeAddEvaluator(**opt['model']['args']).to(device)
    ckpt = torch.load(weight_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    model.eval()
    mean_score = 0.14759505
    std_score = 0.03613452
    for video_path in tqdm(video_paths, desc='FastVQA'):
        reader = decord.VideoReader(str(video_path))
        data_opt = opt['data']['val-kv1k']['args']
        sample_types = data_opt['sample_types']
        vsamples = {}
        for sample_type, sample_args in sample_types.items():
            if data_opt.get('t_frag', 1) > 1:
                sampler = FragmentSampleFrames(fsize_t=sample_args['clip_len'] // sample_args.get('t_frag', 1), fragments_t=sample_args.get('t_frag', 1), num_clips=sample_args.get('num_clips', 1))
            else:
                sampler = SampleFrames(clip_len=sample_args['clip_len'], num_clips=sample_args['num_clips'])
            num_clips = sample_args.get('num_clips', 1)
            indices = sampler(len(reader))
            frame_dict = {idx: reader[idx] for idx in np.unique(indices)}
            video = torch.stack([frame_dict[idx] for idx in indices], 0).permute(3, 0, 1, 2)
            sampled_video = get_spatial_fragments(video, **sample_args)
            mean = torch.FloatTensor([123.675, 116.28, 103.53])
            std = torch.FloatTensor([58.395, 57.12, 57.375])
            sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
            sampled_video = sampled_video.reshape(sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]).transpose(0, 1)
            vsamples[sample_type] = sampled_video.to(device)
        with torch.no_grad():
            raw = model(vsamples).mean().item()
        score = 1 / (1 + np.exp(-((raw - mean_score) / std_score)))
        results[Path(video_path).stem] = {'fastvqa': float(score)}
    return results


def merge_results(base, extra):
    for key, values in extra.items():
        base.setdefault(key, {}).update(values)


def aggregate(results, metric_order=None):
    metric_keys = set()
    for values in results.values():
        metric_keys.update(values.keys())
    if metric_order is None:
        metric_order = sorted(metric_keys)
    else:
        metric_order = [m for m in metric_order if m in metric_keys]
        metric_order += [m for m in sorted(metric_keys) if m not in metric_order]
    average = {}
    for metric in metric_order:
        vals = [values[metric] for values in results.values() if metric in values and values[metric] is not None]
        if vals:
            average[metric] = float(np.mean(vals))
    return {'per_sample': results, 'average': average}


def evaluate_metrics(pred_dir: str, gt_dir: str, output_dir: str, cfg=None):
    cfg = cfg or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    metrics_list = [m.lower() for m in cfg.get('metrics', ALL_METRICS)]

    results = compute_pyiqa_metrics(
        pred_dir,
        gt_dir,
        metrics_list,
        device,
        batch_mode=cfg.get('batch_mode', False),
        crop=cfg.get('crop', 0),
        test_y_channel=cfg.get('test_y_channel', False),
        is_center=cfg.get('is_center', False),
    )
    temp_dir, video_paths = prepare_temp_videos(
        pred_dir,
        output_dir,
        fps=cfg.get('fps', 25),
        save_format=cfg.get('save_format', 'yuv444p'),
    )

    if 'ewarp' in metrics_list:
        merge_results(results, compute_ewarp(video_paths, device))
    if 'dover' in metrics_list:
        merge_results(results, compute_dover(video_paths, device))
    if 'fastvqa' in metrics_list:
        merge_results(results, compute_fastvqa(video_paths, device))

    if 'vbench' in metrics_list:
        try:
            sys.path.insert(0, str(METRICS_ROOT / 'VBench'))
            from evaluate import calculate_final as vbench_eval
            v_results, _, _, _ = vbench_eval(str(temp_dir))
            merge_results(results, {k: {'vbench': float(v)} for k, v in v_results.items()})
        except Exception as exc:
            print(f'Failed to compute VBench: {exc}')

    if 'mdvqa' in metrics_list:
        try:
            mdvqa_path = METRICS_ROOT / 'MD-VQA'
            sys.path.insert(0, str(mdvqa_path))
            from video_quality.video_quality import MDVQA
            if device.type == 'cuda':
                evaluator = MDVQA(
                    use_cudnn=True,
                    semantic_path=str(mdvqa_path / 'data' / 'efficientnet_v2_s-dd5fe13b.pth'),
                    motion_path=str(mdvqa_path / 'data' / 'r2plus1d_18-91a641e6.pth'),
                    mdvqa_path=str(mdvqa_path / 'data' / 'LSVQ_rp0.pth'),
                )
                mdvqa_results = {}
                for vp in tqdm(video_paths, desc='MDVQA'):
                    r = evaluator(video_path=str(vp))
                    mdvqa_results[Path(vp).stem] = {'mdvqa': float(r['score'])}
                merge_results(results, mdvqa_results)
            else:
                print('MDVQA skipped: CUDA required')
        except Exception as exc:
            print(f'Failed to compute MDVQA: {exc}')

    final = aggregate(results, metric_order=metrics_list)
    print('Average metrics:')
    ordered_avg = {m: final['average'][m] for m in metrics_list if m in final['average']}
    print(json.dumps(ordered_avg, indent=2, ensure_ascii=False))
    return final

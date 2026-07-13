import argparse
import json
from pathlib import Path
import sys
sys.path.append('/data2/wujialing/project/STVSR/WanSTVSR')
import torch

from utils.all_metrics import ALL_METRICS, evaluate_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Compute STVSR metrics for generated videos or PNG frame folders.')
    parser.add_argument('--gt', type=str, default='', help='GT folder. Required for full-reference metrics.')
    parser.add_argument('--pred', type=str, required=True, help='Predicted SR folder.')
    parser.add_argument('--out', type=str, default='metrics_results', help='Directory for metric artifacts and JSON.')
    parser.add_argument('--metrics', type=str, default=','.join(ALL_METRICS), help='Comma-separated metric list.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--fps', type=int, default=8)
    parser.add_argument('--filename', type=str, default='all_metrics_results.json')
    parser.add_argument('--batch_mode', action='store_true', help='Use batch mode for PyIQA metrics computation')
    parser.add_argument('--crop', type=int, default=0, help='Crop border size for full-reference metrics')
    parser.add_argument('--test_y_channel', action='store_true', help='Use Y channel for full-reference metrics')
    parser.add_argument('--is_center', action='store_true', help='Use center crop for GT/pred alignment; default is top-left crop')
    args = parser.parse_args()
    args.metrics = [m.strip().lower() for m in args.metrics.split(',') if m.strip()]
    return args


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        'metrics': args.metrics,
        'device': args.device,
        'fps': args.fps,
        'batch_mode': args.batch_mode,
        'crop': args.crop,
        'test_y_channel': args.test_y_channel,
        'is_center': args.is_center,
    }
    results = evaluate_metrics(args.pred, args.gt, str(out_dir), cfg)
    filename = args.filename if args.filename.endswith('.json') else f'{args.filename}.json'
    out_file = out_dir / filename
    out_file.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'Evaluation complete. Results saved to {out_file}')


if __name__ == '__main__':
    main()

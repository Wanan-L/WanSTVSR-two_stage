
import torch
from safetensors.torch import load_file as load_safetensors_file
from pathlib import Path
import json

def _load_single_checkpoint_file(path: Path):
    if path.suffix == ".safetensors":
        if load_safetensors_file is None:
            raise ImportError(
                f"Loading safetensors checkpoint requires `safetensors`: {path}"
            )
        return load_safetensors_file(str(path), device="cpu")

    return torch.load(str(path), map_location='cpu', weights_only=False)


def _load_sharded_checkpoint(index_path: Path):
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) == 0:
        raise RuntimeError(f"Invalid sharded checkpoint index: {index_path}")

    state_dict = {}
    shard_files = sorted(set(weight_map.values()))
    for shard_name in shard_files:
        shard_path = index_path.parent / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard file listed in {index_path}: {shard_path}")

        shard = _load_single_checkpoint_file(shard_path)
        shard_state_dict = shard.get('state_dict', shard) if isinstance(shard, dict) else shard
        if not isinstance(shard_state_dict, dict):
            raise RuntimeError(f"Shard is not a state_dict-like object: {shard_path}")

        for k, v in shard_state_dict.items():
            if isinstance(v, torch.Tensor):
                state_dict[k] = v

    return state_dict


def load_checkpoint_state_dict(ckpt_path):
    path = Path(ckpt_path)

    if path.is_file():
        ckpt = _load_single_checkpoint_file(path)
        return ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

    if path.is_dir():
        pytorch_index = path / "pytorch_model.bin.index.json"
        safetensors_index = path / "model.safetensors.index.json"

        if pytorch_index.exists():
            return _load_sharded_checkpoint(pytorch_index)
        if safetensors_index.exists():
            return _load_sharded_checkpoint(safetensors_index)

        for filename in ("pytorch_model.bin", "model.safetensors"):
            weight_file = path / filename
            if weight_file.exists():
                ckpt = _load_single_checkpoint_file(weight_file)
                return ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

        if (path / "zero_to_fp32.py").exists() or (path / "checkpoint").exists():
            raise RuntimeError(
                "Detected a DeepSpeed ZeRO checkpoint directory. "
                "Convert it first, then pass the converted output directory to "
                "`--wanstvsr_model_path`. Example:\n"
                f"  cd {path}\n"
                "  python zero_to_fp32.py . output_dir --max_shard_size 5GB\n"
                f"  --wanstvsr_model_path {path / 'output_dir'}"
            )

    raise FileNotFoundError(f"Checkpoint path not found or unsupported: {ckpt_path}")
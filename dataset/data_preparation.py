import argparse
import os
import sys
from pathlib import Path

import torch
import lightning as pl
import yaml

sys.path.append("/data2/wujialing/project/STVSR/WanSTVSR")

from dataset.spatio_temporal_real_sr_dataset import SpatioTemporalRealSRDataset as TextVideoDataset
from diffsynth import ModelManager
from diffsynth.pipelines.wan_stvsr import WanSTVSRPipeline

class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, save_path, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [text_encoder_path, vae_path]

        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanSTVSRPipeline.from_model_manager(model_manager)

        self.save_path = Path(save_path)
        self.empty_prompt_embedding_path = self.save_path / "empty_prompt_embedding.pth"
        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

    def on_test_start(self):
        """空文本只编码并保存一次。PyTorch Lightning 会在测试开始时调用。"""
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.pipe.device = self.device

        if self.trainer.is_global_zero and not self.empty_prompt_embedding_path.exists():   # 只在主进程编码一次
            self.pipe.load_models_to_device(["text_encoder"])
            empty_prompt_emb = self.pipe.encode_prompt("")
            torch.save({"prompt_emb": empty_prompt_emb},self.empty_prompt_embedding_path,)

        if self.trainer.world_size > 1:     # 多卡同步
            self.trainer.strategy.barrier()

    def test_step(self, batch, batch_idx):
        path = None
        try:
            self.pipe.device = self.device
            self.pipe.load_models_to_device(["vae"])

            # Dataset 返回 [C, T, H, W]
            video = batch["hq_video"]
            lq_video = batch["lq_video"]
            src_path = batch["path"][0]

            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)  # [B, C, T, H, W]
            lq_video = lq_video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

            # 按 WanSTVSRPipeline.test() 的逻辑先 pad，再 encode_video。
            video, pad_h, pad_w = self.pipe.pad_spatial(video)
            video, pad_t = self.pipe.pad_temporal(video)

            lq_video, lq_pad_h, lq_pad_w = self.pipe.pad_spatial(lq_video)
            lq_video, lq_pad_t = self.pipe.pad_temporal(lq_video)

            latents = self.pipe.encode_video(video, **self.tiler_kwargs).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)[0]     # [B, T, C]
            lq_latents = self.pipe.encode_video(lq_video, **self.tiler_kwargs).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)[0]

            src_stem = Path(src_path).stem      # 去掉路径中的扩展名
            path = self.save_path / f"{src_stem}.tensors.pth"
            # path = self.save_path / f"{batch_idx:08d}_{src_stem}.tensors.pth"
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "video_shape": tuple(batch["hq_video"].shape[1:]),  # [C, T, H, W]
                "latents": latents.detach().cpu(),          # GT latent
                "lq_latents": lq_latents.detach().cpu(),    # LQ latent
                "empty_prompt_embedding_path": str(self.empty_prompt_embedding_path),
                "pad_info": {
                "hq": {"pad_h": pad_h, "pad_w": pad_w, "pad_t": pad_t},
                "lq": {"pad_h": lq_pad_h, "pad_w": lq_pad_w, "pad_t": lq_pad_t},
                },
            }
            torch.save(data, path)

        except Exception as e:
            print(f"[data_preparation] failed at path={path}, error={repr(e)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute WanSTVSR training tensors.")

    parser.add_argument("--output_path", type=str, default="./data_logs", help="Path for Lightning outputs.")
    parser.add_argument("--save_path", type=str, required=True, help="Path for saving tensors.")

    parser.add_argument("--data_root", type=str, required=True, help="Path to the dataset root.")
    parser.add_argument("--degradation_config", type=str, required=True)

    parser.add_argument("--num_frames", type=int, default=17, help="Number of frames in each video.")
    parser.add_argument("--height", type=int, default=320, help="Height of the video.")
    parser.add_argument("--width", type=int, default=640, help="Width of the video.")
    parser.add_argument("--spatial_scale", type=int, default=4, help="Spatial scale factor.")
    parser.add_argument("--temporal_scale", type=int, default=2, help="Temporal scale factor.")
    parser.add_argument("--inter_frame_margin", type=int, default=10, help="Inter-frame margin.")

    parser.add_argument("--text_encoder_path", type=str, required=True, help="Path to the text encoder model.")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to the VAE model.")

    parser.add_argument("--tiled", default=False, action="store_true", help="Whether enable tile encode in VAE. This option can reduce VRAM required.")
    parser.add_argument("--tile_size_height", type=int, default=34, help="Tile size (height) in VAE.")
    parser.add_argument("--tile_size_width", type=int, default=34, help="Tile size (width) in VAE.")
    parser.add_argument("--tile_stride_height", type=int, default=18, help="Tile stride (height) in VAE.")  
    parser.add_argument("--tile_stride_width", type=int, default=16, help="Tile stride (width) in VAE.")

    parser.add_argument("--dataloader_num_workers", type=int, default=1, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")

    return parser.parse_args()


def data_process(args):
    """ 数据处理函数。预先计算 WanSTVSR 视频训练张量。"""
    os.makedirs(args.save_path, exist_ok=True)

    dataset = TextVideoDataset(
        data_root=args.data_root,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        spatial_scale=args.spatial_scale,
        temporal_scale=args.temporal_scale,
        degradation_config=args.degradation_config,
        inter_frame_margin=args.inter_frame_margin,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )

    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        vae_path=args.vae_path,
        save_path=args.save_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)


if __name__ == "__main__":
    args = parse_args()
    data_process(args)

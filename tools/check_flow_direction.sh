CUDA_VISIBLE_DEVICES=6 python tools/check_flow_direction.py \
  --video /data2/wujialing/data/VSR/HQ-VSR/__jqamhpbE4_11_16to287.mp4 \
  --raft_ckpt_path utils/RAFT/raft-things.pth \
  --num_frames 5 \
  --height 320 \
  --width 160 \
  --device cuda \
  --save_dir ./flow_direction_debug

# check flow direction
(torch2.8) root@7c7e6a5162a9:/data2/wujialing/project/STVSR/WanSTVSR# CUDA_VISIBLE_DEVICES=6 python tools/check_flow_direction.py   --video /data2/wujialing/data/VSR/HQ-VSR/__jqamhpbE4_11_16to287.mp4   --raft_ckpt_path utils/RAFT/raft-things.pth   --num_frames 5   --height 320   --width 160   --device cuda   --save_dir ./flow_direction_debug
pair 00-01: normal=0.016023, swapped=0.029843
pair 01-02: normal=0.013511, swapped=0.024419
pair 02-03: normal=0.014688, swapped=0.022856
pair 03-04: normal=0.014140, swapped=0.019557

=== Flow direction check ===
normal mean error : 0.014591
swapped mean error: 0.024169

Result: current training direction is likely correct.
Keep --swap_flow_directions disabled.

Saved debug images to: ./flow_direction_debug
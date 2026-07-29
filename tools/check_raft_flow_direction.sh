cd /data2/wujialing/project/STVSR/WanSTVSR-0713
CUDA_VISIBLE_DEVICES=1 python tools/check_raft_flow_direction.py \
  --raft_ckpt_path ./utils/RAFT/raft-things.pth

(torch2.8) root@6ce85d3e00bf:/data2/wujialing/project/STVSR/WanSTVSR-0713/tools# cd /data2/wujialing/project/STVSR/WanSTVSR-0713
CUDA_VISIBLE_DEVICES=1 python tools/check_raft_flow_direction.py \
  --raft_ckpt_path ./utils/RAFT/raft-things.pth
[flow_warp one-pixel test]
  horizontal flow -1 max error: 0.00000012
  horizontal flow +1 max error: 1.00000000
  PASS: flow_warp uses backward sampling: output(x)=source(x+flow(x)).

[RAFT_bi output-direction diagnostic]
  synthetic horizontal translation: +8 pixels
  no-warp frame0/frame1 MAE: 0.08228127
  forward: frame0 -> frame1        MAE: 0.08477864
  backward: frame0 -> frame1       MAE: 0.00153887
  forward: frame1 -> frame0        MAE: 0.00128755
  backward: frame1 -> frame0       MAE: 0.08309806

[recommended TE setting]
  FLOW_TO_NEXT_OUTPUT = "backward"
  The selected output reduced the frame0 -> frame1 warp error.
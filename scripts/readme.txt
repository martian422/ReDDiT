如何运行训练？

0.环境配置好（cuda=12.1, python=3.10 + requirements）。启动该环境。
1.在configs/data/**.yaml中配置好dataset_path和image_token_dir的地址。
  如果使用repa_loss，还需要在config里配置dinov2地址。
2.进入当前目录（如/nfs/code/ddit-c2i/）
3.对单卡训练：运行形如bash scripts/llamagen/train_ddit.sh 的命令即可。
  对多卡训练：在拥有共享存储的两台机器上分别运行
  bash scripts/llamagen/train_ddit_multi.sh 0 (主机)
  bash scripts/llamagen/train_ddit_multi.sh 1 （从属）
  多于两台机器时依次增加数字，记得修改.sh里的NNODES和trainer.num_nodes。
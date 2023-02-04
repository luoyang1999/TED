#!/usr/bin/env bash


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=8 train.py --cfg_file cfgs/models/dair_v2x/TED-S-inf.yaml --launcher pytorch > log.txt&

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=8 train.py --cfg_file cfgs/models/dair_v2x/TED-S-early.yaml --launcher pytorch

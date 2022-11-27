# CUDA_VISIBLE_DEVICES=0,1,2 python imagenet_quant.py -a resnet50  ./tiny-imagenet-200 -s 0.01 -w -g --smtk 0

CUDA_VISIBLE_DEVICES=7 python imagenet_quant.py -s 0.1 -w -g --smtk 0 --start-subset 0 --subset_schedule cnt &
CUDA_VISIBLE_DEVICES=6 python imagenet_quant.py -s 0.2 -w -g --smtk 0 --start-subset 0 --subset_schedule cnt &
CUDA_VISIBLE_DEVICES=5 python imagenet_quant.py -s 0.3 -w -g --smtk 0 --start-subset 0 --subset_schedule cnt &
CUDA_VISIBLE_DEVICES=4 python imagenet_quant.py -s 0.4 -w -g --smtk 0 --start-subset 0 --subset_schedule cnt &
CUDA_VISIBLE_DEVICES=3 python imagenet_quant.py -s 0.5 -w -g --smtk 0 --start-subset 0 --subset_schedule cnt &
CUDA_VISIBLE_DEVICES=2 python imagenet_quant.py -s 0.5 -w -g --smtk 0 --start-subset 30 --subset_schedule step &
CUDA_VISIBLE_DEVICES=1 python imagenet_quant.py -s 0.1 -w -g --smtk 0 --start-subset 50 --subset_schedule cnt &
CUDA_VISIBLE_DEVICES=0 python imagenet_quant.py -s 0.02 -w -g --smtk 0 --start-subset 60 --subset_schedule cnt &
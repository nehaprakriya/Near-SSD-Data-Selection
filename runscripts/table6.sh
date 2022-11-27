# Training on the whole dataset
# python3 train_resnet.py --dataset cifar10 --arch resnet20 --smtk 0 --gpu 0

# # Vanilla
# python3 train_resnet.py --dataset cifar10 --arch resnet20 --smtk 0 -s 0.1 -g --gpu 0
# python3 train_resnet.py --dataset cifar10 --arch resnet20 --smtk 0 -s 0.3 -g --gpu 0
# python3 train_resnet.py --dataset cifar10 --arch resnet20 --smtk 0 -s 0.5 -g --gpu 0

# # With Partition (PA)
# python3 train_resnet.py --dataset cifar10 --arch resnet20 --smtk 0 -s 0.1 -g --gpu 0 --partition
# python3 train_resnet.py --dataset cifar10 --arch resnet20 --smtk 0 -s 0.3 -g --gpu 0 --partition
# python3 train_resnet.py --dataset cifar10 --arch resnet20 --smtk 0 -s 0.5 -g --gpu 0 --partition

# # With SS
# python3 train_resnet.py --dataset cifar10 --arch resnet20 --smtk 0 -s 0.1 -g --gpu 0 --drop_learned --drop_thresh 1.4
# python3 train_resnet.py --dataset cifar10 --arch resnet20 --smtk 0 -s 0.3 -g --gpu 0 --drop_learned --drop_thresh 1.4
# python3 train_resnet.py --dataset cifar10 --arch resnet20 --smtk 0 -s 0.5 -g --gpu 0 --drop_learned --drop_thresh 1.4

# With SS + PA 
python3 train_resnet.py --dataset cifar10 --arch resnet20 --smtk 0 -s 0.1 -g --gpu 0 --drop_learned --drop_thresh 1.4 --partition
python3 train_resnet.py --dataset cifar10 --arch resnet20 --smtk 0 -s 0.3 -g --gpu 0 --drop_learned --drop_thresh 1.4 --partition
python3 train_resnet.py --dataset cifar10 --arch resnet20 --smtk 0 -s 0.5 -g --gpu 0 --drop_learned --drop_thresh 1.4 --partition
# python train_resnet.py --dataset cifar10 --arch resnet20 --smtk 0 --gpu 7 --save_stats &
# python train_resnet.py --dataset cifar100 --arch resnet18 --smtk 0 --gpu 6 --save_stats &
# python train_resnet.py --dataset cinic10 --arch resnet18 --smtk 0 --gpu 5 --save_stats &
# python train_resnet.py --dataset svhn --arch resnet18 --smtk 0 --gpu 4 --save_stats &

# python train_resnet.py --dataset cifar10 --arch resnet20 --smtk 0 --gpu 3 --save_stats &
# python train_resnet.py --dataset cifar100 --arch resnet18 --smtk 0 --gpu 2 --save_stats &
# python train_resnet.py --dataset cinic10 --arch resnet18 --smtk 0 --gpu 1 --save_stats &
# python train_resnet.py --dataset svhn --arch resnet18 --smtk 0 --gpu 0 --save_stats &

# python train_resnet.py --dataset cinic10 --arch resnet18 --smtk 0 --gpu 7 --save_stats &
# python train_resnet.py --dataset cinic10 --arch resnet18 --smtk 0 --gpu 6 --save_stats &
# python train_resnet.py --dataset cinic10 --arch resnet18 --smtk 0 --gpu 5 --save_stats &
# python train_resnet.py --dataset cinic10 --arch resnet18 --smtk 0 --gpu 3 --save_stats &

python train_resnet.py --dataset tinyimagenet --arch resnet18 --smtk 0 --gpu 7 --save_stats &
sleep 1
python train_resnet.py --dataset tinyimagenet --arch resnet18 --smtk 0 --gpu 6 --save_stats &
sleep 1
python train_resnet.py --dataset tinyimagenet --arch resnet18 --smtk 0 --gpu 5 --save_stats &
sleep 1
python train_resnet.py --dataset tinyimagenet --arch resnet18 --smtk 0 --gpu 4 --save_stats &
sleep 1
python train_resnet.py --dataset tinyimagenet --arch resnet18 --smtk 0 --gpu 3 --save_stats &
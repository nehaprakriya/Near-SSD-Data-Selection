# python train_resnet.py --save-dir outputs_subset --dataset cifar10 --arch resnet20 -s 0.03 --smtk 0 --gpu 7 --start-subset 60 --subset_schedule cnt --partition & 
# python train_resnet.py --save-dir outputs_subset --dataset cifar10 --arch resnet20 -s 0.02 --smtk 0 --gpu 6 --start-subset 60 --subset_schedule cnt --partition & 
# python train_resnet.py --save-dir outputs_subset --dataset cifar10 --arch resnet20 -s 0.01 --smtk 0 --gpu 5 --start-subset 60 --subset_schedule cnt --partition & 
# python train_resnet.py --save-dir outputs_subset --dataset cifar10 --arch resnet20 -s 0.03 --smtk 0 --gpu 4 --start-subset 60 --subset_schedule cnt --partition -g & 
# python train_resnet.py --save-dir outputs_subset --dataset cifar10 --arch resnet20 -s 0.02 --smtk 0 --gpu 3 --start-subset 60 --subset_schedule cnt --partition -g & 
# python train_resnet.py --save-dir outputs_subset --dataset cifar10 --arch resnet20 -s 0.01 --smtk 0 --gpu 2 --start-subset 60 --subset_schedule cnt --partition -g & 

python train_resnet.py --save-dir outputs_subset --dataset cifar10 --arch resnet20 --smtk 0 --gpu 6 --partition -g --drop_learned --drop_thresh 2 & 
python train_resnet.py --save-dir outputs_subset --dataset cifar100 --arch resnet18 --smtk 0 --gpu 7 --partition -g --drop_learned --drop_thresh 4 & 
python train_resnet.py --save-dir outputs_subset --dataset cinic10 --arch resnet18 --smtk 0 --gpu 6 --partition -g --drop_learned --drop_thresh 2
python train_resnet.py --save-dir outputs_subset --dataset svhn --arch resnet18 --smtk 0 --gpu 4 --partition -g --drop_learned --drop_thresh 2
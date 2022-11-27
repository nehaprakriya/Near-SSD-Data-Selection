# python train_resnet.py -s 0.1 -w -b 512 -g --smtk 0 --gpu 7 --start-subset 0 --subset_schedule cnt --partition &
# python train_resnet.py -s 0.2 -w -b 512 -g --smtk 0 --gpu 6 --start-subset 0 --subset_schedule cnt --partition &
# python train_resnet.py -s 0.3 -w -b 512 -g --smtk 0 --gpu 5 --start-subset 0 --subset_schedule cnt --partition &
# python train_resnet.py -s 0.4 -w -b 512 -g --smtk 0 --gpu 4 --start-subset 0 --subset_schedule cnt --partition &
# python train_resnet.py -s 0.5 -w -b 512 -g --smtk 0 --gpu 3 --start-subset 0 --subset_schedule cnt --partition &
# python train_resnet.py -s 0.5 -w -b 512 -g --smtk 0 --gpu 2 --start-subset 30 --subset_schedule step --partition &
# python train_resnet.py -s 0.1 -w -b 512 -g --smtk 0 --gpu 1 --start-subset 50 --subset_schedule cnt --partition &
# python train_resnet.py -s 0.02 -w -b 512 -g --smtk 0 --gpu 3 --start-subset 60 --subset_schedule cnt --partition &

# python train_resnet.py -s 0.5 -w -b 512 -g --smtk 0 --gpu 6 --start-subset 30 --subset_schedule step &
# python train_resnet.py -s 0.1 -w -b 512 -g --smtk 0 --gpu 7 --start-subset 50 --subset_schedule cnt &
# python train_resnet.py -s 0.02 -w -b 512 -g --smtk 0 --gpu 3 --start-subset 60 --subset_schedule cnt &

# python train_resnet.py -s 0.5 -w -b 512 -g --smtk 0 --gpu 0 --start-subset 30 --subset_schedule step --dataset cifar100 &
# python train_resnet.py -s 0.1 -w -b 512 -g --smtk 0 --gpu 1 --start-subset 50 --subset_schedule cnt --dataset cifar100 &
# python train_resnet.py -s 0.02 -w -b 512 -g --smtk 0 --gpu 2 --start-subset 60 --subset_schedule cnt --dataset cifar100 &

# python train_resnet.py -s 0.05 -w -b 512 -g --smtk 0 --gpu 0 --start-subset 200 --subset_schedule reduce --dataset cifar100 &
# python train_resnet.py -s 0.02 -w -b 512 -g --smtk 0 --gpu 1 --start-subset 200 --subset_schedule reduce --dataset cifar100 &
# python train_resnet.py -s 0.01 -w -b 512 -g --smtk 0 --gpu 2 --start-subset 200 --subset_schedule reduce --dataset cifar100 &

# python train_resnet.py -s 0.05 -w -b 512 -g --smtk 0 --gpu 0 --start-subset 200 --subset_schedule reduce --dataset cifar100 --lr_schedule reduce &
# python train_resnet.py -s 0.02 -w -b 512 -g --smtk 0 --gpu 1 --start-subset 200 --subset_schedule reduce --dataset cifar100 --lr_schedule reduce &
# python train_resnet.py -s 0.01 -w -b 512 -g --smtk 0 --gpu 2 --start-subset 200 --subset_schedule reduce --dataset cifar100 --lr_schedule reduce &

python train_resnet.py -s 0.1 -w -b 512 --smtk 0 --gpu 7 --start-subset 0 --subset_schedule cnt --partition &
python train_resnet.py -s 0.2 -w -b 512 --smtk 0 --gpu 6 --start-subset 0 --subset_schedule cnt --partition &
python train_resnet.py -s 0.3 -w -b 512 --smtk 0 --gpu 5 --start-subset 0 --subset_schedule cnt --partition &
python train_resnet.py -s 0.4 -w -b 512 --smtk 0 --gpu 4 --start-subset 0 --subset_schedule cnt --partition &
python train_resnet.py -s 0.5 -w -b 512 --smtk 0 --gpu 3 --start-subset 0 --subset_schedule cnt --partition &
python train_resnet.py -s 0.5 -w -b 512 --smtk 0 --gpu 2 --start-subset 30 --subset_schedule step --partition &
python train_resnet.py -s 0.1 -w -b 512 --smtk 0 --gpu 1 --start-subset 50 --subset_schedule cnt --partition &
python train_resnet.py -s 0.02 -w -b 512 --smtk 0 --gpu 0 --start-subset 60 --subset_schedule cnt --partition &
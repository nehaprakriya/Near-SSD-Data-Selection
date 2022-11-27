#!/usr/bin/env python3
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from resnet50_quant_gpu import resnet50
import datetime

import util
import numpy as np
from torch.utils.data import DataLoader
torch.cuda.empty_cache()
#  CUDA_VISIBLE_DEVICES=1,2 python imagenet_icml.py --dist-url 'tcp://127.0.0.1:8887' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0  /lfs/local/0/baharanm/data/imagenet -s 0.3 -g --smtk 2 -w -b 512
#  python imagenet_icml.py -a resnet18  /lfs/local/0/baharanm/data/imagenet -s 0.01 -w -g --smtk 1
#pip uninstall numpy,  pip install numpy==1.16.4
# os.environ["CUDA_VISIBLE_DEVICES"] = "5,6" # TODO

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
print(model_names)
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='./tiny-imagenet-200',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='outputs', type=str)
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--greedy', '-g', dest='greedy', action='store_true', default=False, help='greedy ordering')
parser.add_argument('--subset_size', '-s', dest='subset_size', type=float, help='size of the subset', default=0.1)
parser.add_argument('--st_grd', '-stg', type=float, help='stochastic greedy', default=0)
parser.add_argument('--smtk', type=int, help='smtk', default=1)
parser.add_argument('--ig', type=str, help='ig method', default='sgd', choices=['sgd, adam, adagrad'])
parser.add_argument('--warm', '-w', dest='warm_start', action='store_true', default=False, help='warm start learning rate ')
parser.add_argument('--runs', type=int, help='num runs', default=3)
parser.add_argument('--lag', type=int, help='update lags', default=1)
parser.add_argument('--cluster_features', '-cf', dest='cluster_features', action='store_true', help='cluster_features')
parser.add_argument('--cluster_all', '-ca', dest='cluster_all', action='store_true', help='cluster_all')
parser.add_argument('--start-subset', '-st', default=0, type=int, metavar='N', help='start subset selection')
parser.add_argument('--save_subset', dest='save_subset', action='store_true', help='save_subset')
# parser.add_argument('--random_subset_size', '-rs', type=float, help='size of the subset', default=1.0)
parser.add_argument('--no-weight', '-nw', dest='weighted', action='store_false', default=True, help='weighted subset')
parser.add_argument('--test', dest='test', action='store_true', default=False, help='train on val set')
parser.add_argument('--decay_all', '-da', dest='decay_all', action='store_true', default=False, help='use the original weight decay')
parser.add_argument('--subset_schedule', type=str, help='subset size schedule', default='cnt', choices=['cnt', 'step'])

def main():
    args = parser.parse_args()

    grd = 'grd_w' if args.greedy else 'rand'
    grd += f'_st_{args.st_grd}' if args.st_grd > 0 else ''
    grd += f'_warm' if args.warm_start > 0 else ''
    grd += f'_feature' if args.cluster_features else ''
    grd += f'_ca' if args.cluster_all else ''
    folder = f'./{args.save_dir}/tinyimagenet'
    save_path = f'{folder}_{args.ig}_moment_{args.momentum}_{args.arch}_{args.subset_size}_{grd}_start_{args.start_subset}_lag_{args.lag}_{args.subset_schedule}size'
    today = datetime.datetime.now()
    timestamp = today.strftime("%m-%d-%Y-%H:%M:%S")
    args.save_dir = f'{save_path}_{timestamp}'

    # Check the save_dir exists or not
    os.makedirs(args.save_dir)

    args.class_num = 200

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    mps = mp.get_context('spawn')
    queue = mps.Queue()
    ngpus_per_node = torch.cuda.device_count()
    print("ngpus_per_node", ngpus_per_node)
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, queue))
        # p = mps.Process(target=main_worker, args=(args.gpu, ngpus_per_node, args, queue))
        # p.start()
        # p.join()
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, queue)


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def main_worker(gpu, ngpus_per_node, args, queue):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = resnet50(num_classes=200, pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = resnet50(num_classes=200)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        print('Using Distributed DataParallel')
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print('Using DataParallel')
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    train_criterion = nn.CrossEntropyLoss(reduction='none').cuda(args.gpu)  # (Note): args.gpu
    val_criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    decay = args.weight_decay if args.decay_all else 0
    optimizer = torch.optim.SGD(
        model.parameters(), args.lr, momentum=args.momentum, weight_decay=decay)  # (Note): weight_decay=0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.test:
        traindir = os.path.join(args.data, 'val')  # (Note) For quick test
    else:
        traindir = os.path.join(args.data, 'train')

    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)  # (Note): added sampler

    global TRAIN_NUM, CLASS_NUM
    classes = np.unique(train_dataset.targets)
    TRAIN_NUM, CLASS_NUM = len(train_dataset), len(classes)
    assert CLASS_NUM == 200

    global targets
    targets = np.array(train_dataset.targets)

    if args.evaluate:
        validate(val_loader, model, val_criterion, 0, args, queue)
        return

    runs, best_prec1_all = args.runs, 0
    epochs = args.epochs

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
        train_loss, test_loss = np.zeros((runs, epochs)), np.zeros((runs, epochs))
        train_acc, test_acc, test_acc5 = np.zeros((runs, epochs)), np.zeros((runs, epochs)), np.zeros((runs, epochs))
        train_time, data_time = np.zeros((runs, epochs)), np.zeros((runs, epochs))
        grd_time, sim_time, preds_time = np.zeros((runs, epochs)), np.zeros((runs, epochs)), np.zeros((runs, epochs))
        not_selected = np.zeros((runs, epochs))
        times_selected = np.zeros((runs, TRAIN_NUM))

        if args.save_subset:
            B = int(args.subset_size * TRAIN_NUM)
            selected_ndx = np.zeros((runs, epochs, B))
            selected_wgt = np.zeros((runs, epochs, B))

    for run in range(runs):
        '''reset model and optimizer'''
        best_acc1 = 0
        model.apply(weight_reset)
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=0)
        ''''''  # (Note): weight_decay=0

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch, args)
            
            if args.subset_schedule == 'step':
                if epoch < 75:
                    subset_size = args.subset_size
                elif epoch == 75:
                    subset_size = 0.1
                elif epoch == 100:
                    subset_size = 0.01
            else:
                subset_size = args.subset_size

            if subset_size < 1:
                if not args.greedy:
                    B = int(np.sum([int(np.sum(targets == i) * subset_size) for i in classes]))
                    if args.multiprocessing_distributed:
                        B = int(B / args.world_size)
                        num_class_per_gpu = np.ceil(CLASS_NUM / args.world_size)
                        idx = np.where((targets >= args.rank * num_class_per_gpu) &
                                    (targets < (args.rank + 1) * num_class_per_gpu))[0] # todo
                    else:
                        idx = np.arange(0, TRAIN_NUM)
                else:
                    if not args.multiprocessing_distributed:
                        idx = np.arange(0, TRAIN_NUM)
                        # num_class_per_gpu = CLASS_NUM
                    else:
                        num_class_per_gpu = np.ceil(CLASS_NUM / args.world_size)
                        idx = np.where((targets >= args.rank * num_class_per_gpu) &
                                    (targets < (args.rank + 1) * num_class_per_gpu))[0]
                    B = int(subset_size * len(idx))  # (Note) B proportional to len(idx)
                    print(f'[GPU {args.rank}/{args.world_size}] idx: {idx}')

            #######################################################################################
            weight, pred_time = None, 0
            if subset_size >= 1 or epoch < args.start_subset:
                print('Training on all the data')

            elif subset_size < 1 and \
                    (epoch % (args.lag + args.start_subset) == 0 or epoch == args.start_subset):

                if not args.greedy:  # or epoch < args.start_epoch + args.lag: # (Note): warm start
                    print(f'Epoch [{epoch}] [Random] selecting {subset_size * 100}% of {TRAIN_NUM}: '
                          f'[GPU {args.rank}/{args.world_size}]: selecting {B}/{int(subset_size * TRAIN_NUM)}')
                    # order = np.arange(0, TRAIN_NUM)
                    order = idx  # todo
                    # np.random.shuffle(order)  # todo: with replacement
                    # subset, weight = order[:B], None
                    rnd_idx = np.random.randint(0, TRAIN_NUM, B)
                    subset, weight = idx[rnd_idx], None

                    ordering_time, similarity_time, pred_time, fl_labels = 0, 0, 0, 0
                else:
                    if args.cluster_features:
                        print(f'Selecting {B} elements greedily from features')
                        data = datasets.ImageFolder(traindir, transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ]))
                        preds, labels = np.reshape(data.imgs, (TRAIN_NUM, -1)), data.targets
                    else:
                        print(f'Epoch [{epoch}] [Greedy] selecting {subset_size * 100}% of {TRAIN_NUM}: '
                              f'[GPU {args.rank}/{args.world_size}]: selecting {B}/{int(subset_size * TRAIN_NUM)} '
                              f'from class: {np.min(targets[idx])}, '
                              f'to : {np.max(targets[idx])}, with idx: {idx}, labels: {targets[idx]}')

                        q_model_path = os.path.join(args.save_dir, f'resnet50_tinyimagenet_target.pt')
                        torch.save(model.module.state_dict(), q_model_path)
                        print('Size (MB):', os.path.getsize(q_model_path)/1e6)
                        loaded_dict_enc = torch.load(q_model_path, map_location='cpu')
                        q_model = resnet50(num_classes=200, quantize=True)
                        q_model.load_state_dict(loaded_dict_enc)
                        print("loaded state dict")
                        torch.save(q_model.state_dict(), q_model_path)
                        print('Size (MB):', os.path.getsize(q_model_path)/1e6)
                        
                        idx_subset = torch.utils.data.Subset(train_dataset, indices=idx)
                        pred_loader = DataLoader(idx_subset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)  # (Note): shuffle=False
                        preds, pred_time = quantization_predictions(pred_loader, q_model, args, epoch)
                        # preds, pred_time = quantization_predictions(pred_loader, q_model)
                        print(f'Epoch [{epoch}] [Greedy], pred size: {np.shape(preds)}')
                        preds -= np.eye(CLASS_NUM)[targets[idx]]

                    fl_labels = np.zeros(np.shape(targets[idx]), dtype=int) if args.cluster_all else targets[idx] - np.min(targets[idx])
                    subset, weight, _, _, ordering_time, similarity_time = util.get_orders_and_weights(
                        B, preds, 'euclidean', smtk=min(args.rank+2, args.smtk), no=args.smtk, y=fl_labels, stoch_greedy=args.st_grd, equal_num=False)
                    subset = np.array(idx[subset])  # (Note): idx
                    weight = np.array(weight)

                print(f'Epoch [{epoch}] [GPU {args.rank} SENDING] subset: {np.sort(subset)} ({len(subset)}), weight: {weight} '
                      f'({len(weight) if weight is not None else 0}), from class: {np.min(fl_labels)} to class: {np.max(fl_labels)}, '
                      f'order_time: {ordering_time}, '
                      f'sim_time: {similarity_time}, pred_time: {pred_time}')

                if args.multiprocessing_distributed:
                    queue.put({'subset': subset, 'weight': weight, 'order_time': ordering_time,
                               'sim_time': similarity_time, 'pred_time': pred_time})
                # dist.broadcast(tensor=subset_t, group=group, src=0)

                if args.multiprocessing_distributed and args.rank == 0:
                    # Make sure only the first process in distributed training merge the subsets
                    # dist.recv(tensor=subset_t, src=0)
                    subset, weight = [], []
                    ordering_time, similarity_time, pred_time = 0, 0, 0
                    for i in range(args.world_size):
                        msg = queue.get()
                        ordering_time = max(ordering_time, msg['order_time'])
                        similarity_time = max(similarity_time, msg['sim_time'])
                        pred_time = max(pred_time, msg['pred_time'])
                        subset = np.append(subset, msg['subset'])
                        weight = np.append(weight, msg['weight'])
                    subset = subset.astype(int)  # normalize weights

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                    weight = weight / np.sum(weight) * len(subset) if args.weighted and args.greedy else None
                    times_selected[run][subset] += 1
                    not_selected[run, epoch] = np.sum(times_selected[run] == 0) / len(times_selected[run]) * 100
                    grd_time[run, epoch], sim_time[run, epoch], preds_time[run, epoch] = ordering_time, similarity_time, pred_time
                    print(f'Epoch [{epoch}] [GPU {args.rank}] MERGED subsets: {np.sort(subset)} ([{len(subset)}]), (({len(np.unique(subset))})) ' 
                          f'weight: {weight} ({len(weight) if weight is not None else 0}), order_time: {ordering_time},'
                          f' sim_time: {similarity_time}, pred_time: {pred_time}, '
                          f'{not_selected[run, epoch]:.3f} % not selected yet')

                    if args.save_subset:
                        selected_ndx[run, epoch], selected_wgt[run, epoch] = subset, weight

                    if args.multiprocessing_distributed and args.rank == 0:
                        # (Note): randomized order
                        indices = np.array(range(len(subset)))
                        np.random.shuffle(indices)
                        subset = subset[indices]
                        weight = weight[indices] if args.greedy and args.weighted else None

                        for i in range(args.world_size):
                            queue.put({'subset': subset, 'weight': weight})
                    # group = dist.new_group(list(range(1, args.world_size)))
                    # dist.broadcast(subset, 0, group)
                    # dist.broadcast(weight, 0, group)

                if args.multiprocessing_distributed:
                    torch.distributed.barrier()
                    msg = queue.get()
                    subset, weight = msg['subset'], msg['weight']
                    indices = list(range(len(subset)))
                    dist_idx = indices[args.rank:len(subset):args.world_size]  # implementing a distributed sampler
                else:
                    dist_idx = np.array(range(len(subset)))
                    np.random.shuffle(dist_idx)  # (Note): randomized order
                print(f'Epoch [{epoch}] [GPU {args.rank}] training on subset: {subset[dist_idx]}')
                indexed_subset = torch.utils.data.Subset(train_dataset, indices=subset[dist_idx])
                weight = torch.from_numpy(weight[dist_idx]).float().cuda() if args.weighted and args.greedy else None
                train_loader = DataLoader(
                    indexed_subset,
                    batch_size=args.batch_size, shuffle=False,  # (Note): shuffle=False
                    num_workers=args.workers, pin_memory=True)

                    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                    # train_loader = torch.utils.data.DataLoader(
                    #     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                    #     num_workers=args.workers, pin_memory=True, sampler=train_sampler)
                    # weight = None
            else:
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                    print(f'Using the previous subset of size: {len(subset)}')
                    not_selected[run, epoch] = not_selected[run, epoch - 1]
                    print(f'{not_selected[run, epoch]:.3f} % not selected yet')
                #######################################################################################

            # train for one epoch
            tr_loss, tr_num, ttime, dtime = train(train_loader, model, train_criterion, optimizer, epoch, args, weight)

            # torch.distributed.barrier()
            def sum_tensor(tensor):
                rt = tensor.clone()
                # dist.all_reduce(rt, op=dist.reduce_op.SUM)
                dist.all_reduce(rt, op=dist.ReduceOp.SUM)
                return rt

            if args.multiprocessing_distributed:
                metrics = torch.tensor([tr_loss, tr_num, ttime, dtime]).float().cuda()
                metrics /= dist.get_world_size()
                tr_loss, tr_num, ttime, dtime = sum_tensor(metrics).cpu().numpy()
                # tr_loss, tr_num, ttime, dtime = dist.reduce(metrics, 0,  op=dist.reduce_op.SUM).cpu().numpy()
                print(f'******* GPU 0 reduced loss: {tr_loss}, {tr_num}, {ttime}, {dtime}')

            # evaluate on training data
            # train_acc[run, epoch], train_loss[run, epoch] = validate(train_val_loader, model, val_criterion, args)

            # evaluate on validation set
            acc1, val_loss, num_val = validate(val_loader, model, val_criterion, epoch, args)

            if args.multiprocessing_distributed:
                metrics = torch.tensor([acc1, acc5, val_loss, num_val]).float().cuda()
                metrics /= dist.get_world_size()
                acc1, acc5, val_loss, num_val = sum_tensor(metrics).cpu().numpy()
                print(f'******* GPU 0 reduced acc1: {acc1}')

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                train_loss[run, epoch], train_time[run, epoch], data_time[run, epoch] = tr_loss, ttime, dtime
                test_acc[run, epoch], test_loss[run, epoch] = acc1, val_loss

                # remember best acc@1 and save checkpoint
                # is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)
                best_prec1_all = max(acc1, best_prec1_all)
                '''
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best)
                '''

                print(f'***************** Run: [{run}], Epoch: [{epoch}], subset_size: {subset_size}, '
                      f'acc1: {acc1}, loss: {train_loss[run, epoch]:.3f}, best_acc1: {best_acc1},  '
                      f'best_acc1_all: {best_prec1_all}, not selected %:{not_selected[run][epoch]}')

                save_path = f'{args.save_dir}/results'

                if args.save_subset:
                    print(
                        f'Saving the results to {save_path}_subset')

                    np.savez(f'{save_path}_subset',
                             train_loss=train_loss, test_acc=test_acc, train_acc=train_acc, test_loss=test_loss,
                             pred_time=preds_time, train_time=train_time, grd_time=grd_time, sim_time=sim_time,
                             not_selected=not_selected, times_selected=times_selected, data_time=data_time,
                             subset=selected_ndx, weights=selected_wgt, test_acc5=test_acc5)
                else:
                    print(
                        f'Saving the results to {save_path}')

                    np.savez(save_path,
                             train_loss=train_loss, test_acc=test_acc, train_acc=train_acc, test_loss=test_loss,
                             pred_time=preds_time, train_time=train_time, grd_time=grd_time, sim_time=sim_time,
                             not_selected=not_selected, times_selected=times_selected, data_time=data_time)
            if args.multiprocessing_distributed:
                torch.distributed.barrier()


def train(train_loader, model, criterion, optimizer, epoch, args, weight=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    si = 0
    end = time.time()
    # for i, (input, target, idx) in enumerate(train_loader):
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = images.size(0)
        batch_slice = slice(si, si + batch_size)
        si += batch_size

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        ce_loss = criterion(output, target)

        if args.decay_all:  # (Note) weight decay is set in optimizer
            loss = ce_loss  # (ce_loss * weight[batch_slice]).mean()  # (Note)
        else:  # (Note) weight decay is set "0" in optimizer
            # reg_loss = 0.5 * args.weight_decay * torch.sum(torch.as_tensor([
            #     torch.sum(w ** 2) for (name, w) in model.named_parameters() if 'bn' not in name]))
            reg_loss = 0
            for name, param in model.named_parameters():
                if 'bn' not in name:
                    reg_loss += torch.norm(param)

            loss = ce_loss + reg_loss * args.weight_decay
        loss = loss.mean() if weight is None else (loss * weight[batch_slice]).mean()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    # queue.put({'loss': losses.avg, 'num': len(train_loader.dataset), 'ttime': batch_time.sum, 'dtime': data_time.sum})
    return losses.avg, len(train_loader.dataset), batch_time.sum, data_time.sum


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix="Test Epoch: [{}]".format(epoch))

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' ******************* Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} ******************* '
              .format(top1=top1, top5=top5))

    # queue.put({'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg, 'num': len(val_loader.dataset)})
    # print(f'validate: {args.rank} returned')
    return top1.avg, losses.avg, len(val_loader.dataset)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        # print("are we calling this one")
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        # print(self.meters)
        # print(type(self.meters))
        # for meter in self.meters:
        #     # prints(type(meter))
        #     # print(meter)
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.warm_start and epoch < 20:
        lr = args.lr / 20 * (epoch + 1)
        print(f'Warm start learning rate: {lr}')
    else:
        lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum().view(-1)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# def quantization_predictions(loader, model):
#     model.to('cpu')
#     model.eval()
#     preds = np.zeros((TRAIN_NUM, CLASS_NUM))
#     labels = np.zeros(TRAIN_NUM)
#     labels=labels.astype('int32')
#     for i, (input, target, idx) in enumerate(loader):
#         # print(model(input).shape)
#         preds[idx, :] = nn.Softmax(dim=1)(model(input))
#         labels[idx] = target.int()
#     return preds, labels


def quantization_predictions(loader, model, args, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(loader), [batch_time],
        prefix="Predict Epoch: [{}]".format(epoch))

    preds = np.zeros((len(loader.dataset), CLASS_NUM))

    # switch to evaluate mode
    model.eval()
    model.cuda()
    end = time.time()
    si = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            # input_var = input.cuda()
            input_var = input.cuda()
            batch_size = input.size(0)
            batch_slice = slice(si, si + batch_size)
            si += batch_size

            logit = model(input_var)
            # print(logit.shape)
            # preds[batch_slice, :] = nn.Softmax(dim=1)(logit)
            # preds[batch_slice, :] = torch.nn.functional.softmax(logit, dim=1).cpu().data.numpy()  # todo: Note
            preds[batch_slice, :] = torch.nn.functional.log_softmax(logit, dim=1).cpu().data.numpy()
            # labels[idx] = target.int()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)

    # return preds.cpu().data.numpy(), labels.cpu().data.numpy(), batch_time.sum
    return preds, batch_time.sum


if __name__ == '__main__':
    main()

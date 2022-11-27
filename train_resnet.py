import argparse
import os
import time
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import util
from warnings import simplefilter
from GradualWarmupScheduler import *

from resnet import resnet20 as target_resnet20
from resnet_quant import resnet20 as quant_resnet20
# from resnet_quant_gpu import resnet18, resnet50

import datetime
from torch.utils.tensorboard import SummaryWriter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
np.seterr(all='ignore')

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', #'resnet56', #
                    help='model architecture: ' +
                         ' (default: resnet18)')
parser.add_argument('--data_dir', default='~/data')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'cinic10', 'svhn', 'tinyimagenet', 'imagenet'],
                    help='dataset: ' + ' (default: cifar10)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', '-m', type=float, metavar='M', default=0.9,
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='outputs', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=300)  # default=10)
parser.add_argument('--gpu', default='0', type=str, help='The GPU to be used')
parser.add_argument('--greedy', '-g', dest='greedy', action='store_true', default=False, help='greedy ordering')
parser.add_argument('--uniform_weight', action='store_true', default=False, help='no sample reweighting')
parser.add_argument('--subset_size', '-s', dest='subset_size', type=float, help='size of the subset', default=1.0)
parser.add_argument('--random_subset_size', '-rs', type=float, help='size of the subset', default=1.0)
parser.add_argument('--st_grd', '-stg', type=float, help='stochastic greedy', default=0)
parser.add_argument('--smtk', type=int, help='smtk', default=1)
parser.add_argument('--ig', type=str, help='ig method', default='sgd', choices=['sgd, adam, adagrad'])
parser.add_argument('--lr_schedule', '-lrs', type=str, help='learning rate schedule', default='mile',
                    choices=['mile', 'exp', 'cnt', 'step', 'cosine', 'reduce'])
parser.add_argument('--gamma', type=float, default=-1, help='learning rate decay parameter')
parser.add_argument('--lag', type=int, help='update lags', default=1)
parser.add_argument('--runs', type=int, help='num runs', default=1)
parser.add_argument('--warm', '-w', dest='warm_start', action='store_true', help='warm start learning rate ')
parser.add_argument('--cluster_features', '-cf', dest='cluster_features', action='store_true', help='cluster_features')
parser.add_argument('--cluster_all', '-ca', dest='cluster_all', action='store_true', help='cluster_all')
parser.add_argument('--start-subset', '-st', default=0, type=int, metavar='N', help='start subset selection')
parser.add_argument('--drop_learned', action='store_true', help='drop learned examples')
parser.add_argument('--watch_interval', default=5, type=int, help='decide whether an example is learned based on how many epochs')
parser.add_argument('--drop_interval', default=20, type=int, help='decide whether an example is learned based on how many epochs')
parser.add_argument('--drop_thresh', default=2, type=float, help='loss threshold')
parser.add_argument('--save_subset', dest='save_subset', action='store_true', help='save_subset')
parser.add_argument('--save_stats', action='store_true', help='save forgetting scores and losses')
parser.add_argument('--partition', dest='partition', action='store_true', help='paritition the dataset by the number of mini-batches')
parser.add_argument('--subset_schedule', type=str, help='subset size schedule', default='cnt',
                    choices=['cnt', 'step', 'reduce'])

def main(args, subset_size=.1, greedy=0):
    global best_prec1
    args = parser.parse_args()

    print(f'--------- subset_size: {subset_size}, method: {args.ig}, moment: {args.momentum}, '
          f'lr_schedule: {args.lr_schedule}, greedy: {greedy}, stoch: {args.st_grd}, rs: {args.random_subset_size} ---------------')
    
    grd = 'grd_w' if args.greedy else f'rand_rsize_{args.random_subset_size}'
    grd += f'_st_{args.st_grd}' if args.st_grd > 0 else ''
    grd += f'_warm' if args.warm_start > 0 else ''
    grd += f'_feature' if args.cluster_features else ''
    grd += f'_ca' if args.cluster_all else ''
    grd += f'_uniform' if args.uniform_weight else ''
    grd += f'_partition' if args.partition else ''
    grd += f'_dropbelow{args.drop_thresh}_every{args.drop_interval}epochs_watch{args.watch_interval}epochs' if args.drop_learned else ''
    folder = f'./{args.save_dir}/{args.dataset}'
    save_path = f'{folder}/{args.ig}_moment_{args.momentum}_{args.arch}_{args.subset_size}_{grd}_{args.lr_schedule}_start_{args.start_subset}_lag_{args.lag}_{args.subset_schedule}size'
    today = datetime.datetime.now()
    timestamp = today.strftime("%m-%d-%Y-%H:%M:%S")
    args.save_dir = f'{save_path}_{timestamp}'
    
    # Check the save_dir exists or not
    os.makedirs(args.save_dir)
    os.makedirs(os.path.join(args.save_dir, 'images'))

    args.writer = SummaryWriter(args.save_dir)

    if args.dataset == 'cifar100':
        args.class_num = 100
    elif args.dataset == 'imagenet':
        args.class_num = 1000
    elif args.dataset == 'tinyimagenet':
        args.class_num = 200
    else:
        args.class_num = 10

    if args.arch == 'resnet20':
        model = target_resnet20(num_classes=args.class_num)
    elif args.arch == 'resnet50':
        model = torch.nn.DataParallel(resnet50(num_classes=args.class_num, cifar=True))
    else:
        model = resnet18(num_classes=args.class_num, cifar=True)
    device='cuda'
    model.to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    class IndexedDataset(Dataset):
        def __init__(self, args):
            self.dataset = util.get_dataset(args)

        def __getitem__(self, index):
            data, target = self.dataset[index]
            return data, target, index

        def __len__(self):
            return len(self.dataset)

    indexed_dataset = IndexedDataset(args)
    indexed_loader = DataLoader(
        indexed_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        util.get_dataset(args, train=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    train_val_loader = torch.utils.data.DataLoader(
        util.get_dataset(args, train=True, train_transform=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    train_criterion = nn.CrossEntropyLoss(reduction='none').cuda()  # (Note)
    val_criterion = nn.CrossEntropyLoss().cuda()

    args.train_num = len(indexed_dataset)

    if args.half:
        model.half()
        train_criterion.half()
        val_criterion.half()

    runs, best_run, best_run_loss, best_loss = args.runs, 0, 0, 1e10
    epochs = args.epochs
    train_loss, test_loss = np.zeros((runs, epochs)), np.zeros((runs, epochs))
    train_acc, test_acc = np.zeros((runs, epochs)), np.zeros((runs, epochs))
    train_time, data_time = np.zeros((runs, epochs)), np.zeros((runs, epochs))
    grd_time, sim_time = np.zeros((runs, epochs)), np.zeros((runs, epochs))
    not_selected = np.zeros((runs, epochs))
    best_bs, best_gs = np.zeros(runs), np.zeros(runs)
    times_selected = np.zeros((runs, len(indexed_loader.dataset)))

    if args.save_subset:
        B = int(args.subset_size * args.train_num)
        selected_ndx = np.zeros((runs, epochs, B))
        selected_wgt = np.zeros((runs, epochs, B))

    if (args.lr_schedule == 'mile' or args.lr_schedule == 'cosine') and args.gamma == -1:
        lr = args.lr
        b = 0.1
    else:
        lr = args.lr
        b = args.gamma

    print(f'lr schedule: {args.lr_schedule}, epochs: {args.epochs}')
    print(f'lr: {lr}, b: {b}')
    order = np.arange(0, args.train_num)
    targets = np.array(indexed_dataset.dataset.targets)

    for run in range(runs):
        best_prec1_all, best_loss_all, prec1 = 0, 1e10, 0
        forgets = np.zeros(args.train_num)
        learned = np.zeros(args.train_num)
        watch = np.zeros((args.watch_interval, args.train_num))

        if subset_size < 1:
            # initialize a random subset
            B = int(args.random_subset_size * args.train_num)
            order = np.arange(0, args.train_num)
            np.random.shuffle(order)
            order = order[:B]
            print(f'Random init subset size: {args.random_subset_size*100}% = {B}')

        if args.arch == 'resnet20':
            model = target_resnet20(num_classes=args.class_num)   
        elif args.arch == 'resnet50':
            model = torch.nn.DataParallel(resnet50(num_classes=args.class_num, cifar=False))         
        else:
            if args.dataset == 'tinyimagenet':
                model = resnet18(num_classes=args.class_num, cifar=False)
            else:
                model = resnet18(num_classes=args.class_num, cifar=True)
        model.cuda()
        
        best_prec1, best_loss = 0, 1e10
        if args.ig == 'adam':
            print('using adam')
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)
        elif args.ig == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)

        if args.lr_schedule == 'exp':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=b, last_epoch=args.start_epoch - 1)
        elif args.lr_schedule == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=b)
        elif args.lr_schedule == 'mile':
            milestones = np.array([60, 120, 160])
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, last_epoch=args.start_epoch - 1, gamma=0.2)
        elif args.lr_schedule == 'cosine':
            # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        elif args.lr_schedule == 'reduce':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
        else:  # constant lr
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs, gamma=1.0)

        if args.warm_start:
            print('Warm start learning rate')
            lr_scheduler_f = GradualWarmupScheduler(optimizer, 1.0, 20, lr_scheduler)
        else:
            print('No Warm start')
            lr_scheduler_f = lr_scheduler

        if args.arch in ['resnet1202', 'resnet110']:
            # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
            # then switch back. In this setup it will correspond for first epoch.
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr*0.1

        if args.evaluate:
            validate(val_loader, model, val_criterion)
            return

        for epoch in range(args.start_epoch, args.epochs):

            curr_lr = optimizer.param_groups[0]['lr']

            # train for one epoch
            print('current lr {:.5e}'.format(curr_lr))

            corrects = np.zeros(args.train_num)
            losses = np.zeros(args.train_num)

            if args.drop_learned and (epoch > 0):
                if (epoch % args.drop_interval == 0) and (len(order) > 1000):
                    order_ = np.where(np.sum(watch>args.drop_thresh, axis=0)>0)[0]
                    if len(order_) > 1000:
                        order = order_
                    subset_size = 1 / args.watch_interval
            elif epoch < args.start_subset:
                subset_size = 1
            elif args.subset_schedule == 'step':
                if epoch < 75:
                    subset_size = args.subset_size
                elif epoch == 75:
                    subset_size = 0.1
                elif epoch == 100:
                    subset_size = 0.01
            else:
                subset_size = args.subset_size

            B = int(subset_size * len(order))
            print(f'Training size at epoch {epoch}: {subset_size*100}% = {B}')
            
            if args.partition and (subset_size < 1) and (epoch >= args.start_subset):
                # random partition the dataset
                partition = int(math.ceil(B / args.batch_size))
                B = min(args.batch_size, int(subset_size * len(order)))
            else:
                partition = 1

            #############################
            weight = None
            for i in range(partition):
                print(f'Training on partition {i+1}/{partition}')
                
                if subset_size >= 1 or epoch < args.start_subset:
                    print('Training on all the data')
                    train_loader = indexed_loader
                    times_selected[run][order] += 1

                    if args.save_stats or args.drop_learned:
                        preds, labels = predictions(args, indexed_loader, model)
                        corrects = np.equal(np.argmax(preds, axis=1), labels)
                        losses = train_criterion(torch.from_numpy(preds), torch.from_numpy(labels).long()).numpy()
                else:
                    if (epoch  % args.lag == 0):
                        q_model_path = os.path.join(args.save_dir, f'{args.dataset}_target.pt')
                        if args.arch == 'resnet50':
                            torch.save(model.module.state_dict(), q_model_path)
                        else:
                            torch.save(model.state_dict(), q_model_path)
                        print('Size (MB):', os.path.getsize(q_model_path)/1e6)
                        loaded_dict_enc = torch.load(q_model_path, map_location='cpu')
                        if args.arch == 'resnet20':
                            q_model = quant_resnet20(num_classes=args.class_num)
                            q_model.load_state_dict(loaded_dict_enc)
                            q_model.to('cpu')
                            q_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                            torch.quantization.prepare(q_model, inplace=True)
                            q_model.eval()
                            torch.quantization.convert(q_model, inplace=True)
                        else:
                            if args.arch == 'resnet50':
                                q_model = resnet50(num_classes=args.class_num, cifar=False, quantize=True)
                            else:
                                if args.dataset == 'tinyimagenet':
                                    q_model = resnet18(num_classes=args.class_num, cifar=False, quantize=True)
                                else:
                                    q_model = resnet18(num_classes=args.class_num, cifar=True, quantize=True)
                            q_model.load_state_dict(loaded_dict_enc)
                            q_model.cuda()
                            q_model.eval()
                        print("loaded state dict")
                        torch.save(q_model.state_dict(), q_model_path)
                        print('Size (MB):', os.path.getsize(q_model_path)/1e6)
                    
                        if args.partition:
                            indices = []
                            num_per_class = int(np.ceil(len(order) / max((len(order) * subset_size / args.batch_size), 1) / args.class_num))
                            _, counts = np.unique(targets[order], return_counts=True)
                            num_per_class = min(np.amin(counts), num_per_class)
                            print(f'Sampling a partition with {num_per_class} examples per class...')
                            for c in np.unique(targets):
                                class_indices = np.intersect1d(np.where(targets == c)[0], order)
                                if num_per_class == len(class_indices):
                                    indices.append(class_indices)
                                else:
                                    indices_per_class = np.random.choice(class_indices, size=num_per_class, replace=False)
                                    indices.append(indices_per_class)
                            indices = np.concatenate(indices)

                            indexed_subset = torch.utils.data.Subset(indexed_dataset, indices=indices)
                            indexed_loader = DataLoader(
                                indexed_subset,
                                batch_size=len(indexed_subset),
                                num_workers=args.workers,
                                pin_memory=True,
                            )
                        else:
                            indices = order
                        if greedy == 0:
                            # order = np.arange(0, TRAIN_NUM)
                            np.random.shuffle(indices)
                            subset = indices[:B]
                            weights = np.zeros(args.train_num)
                            weights[subset] = np.ones(B)
                            print(f'Selecting {B} element from the pre-selected random subset of size: {len(indices)}')
                            weight = torch.from_numpy(weights).float().cuda()
                        else:  # Note: warm start
                            if args.cluster_features:
                                print(f'Selecting {B} elements greedily from features')
                                data = util.get_dataset(args, train=True)
                                preds, labels = np.reshape(data.data, (len(data.targets), -1)), data.targets
                            else:
                                print(f'Selecting {B} elements greedily from predictions')
                                if args.arch == 'resnet20':
                                    preds, labels = quantization_predictions(args, indexed_loader, q_model)
                                else:
                                    preds, labels = predictions(args, indexed_loader, q_model)
                                preds = preds[indices]
                                labels = labels[indices]
                                corrects[indices] = np.equal(np.argmax(preds, axis=1), labels)
                                losses[indices] = train_criterion(torch.from_numpy(preds), torch.from_numpy(labels).long()).numpy()
                                preds -= np.eye(args.class_num)[labels]
                            fl_labels = np.zeros(np.shape(labels), dtype=int) if args.cluster_all else labels
                            subset, subset_weight, _, _, ordering_time, similarity_time = util.get_orders_and_weights(
                                B, preds, 'euclidean', smtk=args.smtk, no=0, y=fl_labels, stoch_greedy=args.st_grd,
                                equal_num=True)
                            subset = indices[subset]

                            if args.uniform_weight:
                                weights = np.zeros(args.train_num)
                                weights[subset] = np.ones(len(subset))
                            else:
                                plt_weights = subset_weight
                                plt_weights[np.where(plt_weights>2*int(1./subset_size))] = 2*int(1./subset_size)
                                fig = plt.figure()
                                plt.hist(plt_weights, bins=np.arange(np.amax(plt_weights)), edgecolor='black')
                                args.writer.add_figure('cluster_weights', fig, epoch)
                                plt.savefig(os.path.join(args.save_dir, f'images/weights_epoch{epoch}.png'))
                                plt.close()
                        
                                weights = np.zeros(args.train_num)
                                subset_weight = subset_weight / np.sum(subset_weight) * len(subset_weight)
                                if args.save_subset:
                                    selected_ndx[run, epoch], selected_wgt[run, epoch] = subset, subset_weight
                                
                                weights[subset] = subset_weight
                            weight = torch.from_numpy(weights).float().cuda()

                            print(f'FL time: {ordering_time:.3f}, Sim time: {similarity_time:.3f}')
                            grd_time[run, epoch], sim_time[run, epoch] = ordering_time, similarity_time

                        times_selected[run][subset] += 1
                        print(f'{np.sum(times_selected[run] == 0) / len(times_selected[run]) * 100:.3f} % not selected yet')
                        not_selected[run, epoch] = np.sum(times_selected[run] == 0) / len(times_selected[run]) * 100
                        indexed_subset = torch.utils.data.Subset(indexed_dataset, indices=subset)
                        if args.partition:
                            train_loader = DataLoader(
                                indexed_subset,
                                batch_size=len(subset), shuffle=True,
                                num_workers=args.workers, pin_memory=True)
                        else:
                            train_loader = DataLoader(
                                indexed_subset,
                                batch_size=args.batch_size, shuffle=True,
                                num_workers=args.workers, pin_memory=True)
                    else:
                        print('Using the previous subset')
                        not_selected[run, epoch] = not_selected[run, epoch - 1]
                        times_selected[run][subset] += 1
                        print(f'{not_selected[run, epoch]:.3f} % not selected yet')
                        #############################

                prec1, loss, data_time_batch, train_time_batch = train(
                    train_loader, model, epoch, train_criterion, optimizer, weight)

                data_time[run, epoch] += data_time_batch
                train_time[run, epoch] += train_time_batch

            args.writer.add_scalar('train/3.train_size', int(len(order)*subset_size), epoch)
            args.writer.add_scalar('train/4.train_frac', np.sum(times_selected[run])/args.train_num/(epoch+1), epoch)

            # evaluate on validation set
            prec1, loss = validate(train_val_loader, model, val_criterion)
            args.writer.add_scalar('train/1.train_loss', loss, epoch)
            args.writer.add_scalar('train/2.train_acc', prec1, epoch)

            # evaluate on validation set
            prec1, loss = validate(val_loader, model, val_criterion)

            if args.lr_schedule == 'reduce':
                lr_scheduler_f.step(loss)
            else:
                lr_scheduler_f.step()

            args.writer.add_scalar('val/1.val_loss', loss, epoch)
            args.writer.add_scalar('val/2.val_acc', prec1, epoch)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            # best_run = run if is_best else best_run
            best_prec1 = max(prec1, best_prec1)
            if best_prec1 > best_prec1_all:
                best_gs[run], best_bs[run] = lr, b
                best_prec1_all = best_prec1
            test_acc[run, epoch], test_loss[run, epoch] = prec1, loss

            args.writer.add_scalar('test/1.test_loss', loss, epoch)
            args.writer.add_scalar('test/2.test_acc', prec1, epoch)

            ta, tl = validate(train_val_loader, model, val_criterion)
            # best_run_loss = run if tl < best_loss else best_run_loss
            best_loss = min(tl, best_loss)
            best_loss_all = min(best_loss_all, best_loss)
            train_acc[run, epoch], train_loss[run, epoch] = ta, tl

            if args.save_stats or args.drop_learned:
                watch[epoch%args.watch_interval] = losses
                if epoch > 0:
                    forgets[learned > corrects] += 1
                    learned = corrects
                
                if (((epoch + 1) % 5) == 0) and args.save_stats:
                    np.save(file=os.path.join(args.save_dir, f'forget_epoch{epoch}.npy'), arr=forgets)
                    fig = plt.figure()
                    plt.hist(forgets, bins=np.arange(np.amax(forgets)+1), edgecolor='black')
                    args.writer.add_figure('forgetting_scores', fig, epoch)
                    plt.hist(forgets, bins=np.arange(np.amax(forgets)), edgecolor='black')
                    plt.savefig(os.path.join(args.save_dir, f'images/forgetting_scores_epoch{epoch}.png'))
                    plt.close()

                    np.save(file=os.path.join(args.save_dir, f'loss_epoch{epoch}.npy'), arr=losses)
                    fig = plt.figure()
                    plt.hist(losses, edgecolor='black')
                    args.writer.add_figure('example_losses', fig, epoch)
                    plt.hist(losses, edgecolor='black')
                    plt.savefig(os.path.join(args.save_dir, f'images/example_losses_epoch{epoch}.png'))
                    plt.close()

            if epoch > 0 and epoch % args.save_every == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

            # save_checkpoint({
            # 'state_dict': model.state_dict(),
            # 'best_prec1': best_prec1,
            # }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

            print(f'run: {run}, subset_size: {subset_size}, epoch: {epoch}, prec1: {prec1}, loss: {tl:.3f}, '
                  f'g: {lr:.3f}, b: {b:.3f}, '
                  f'best_prec1_gb: {best_prec1}, best_loss_gb: {best_loss:.3f}, best_run: {best_run};  '
                  f'best_prec_all: {best_prec1_all}, best_loss_all: {best_loss_all:.3f}, '
                  f'best_g: {best_gs[run]:.3f}, best_b: {best_bs[run]:.3f}, '
                  f'not selected %:{not_selected[run][epoch]}')

            save_path = f'{args.save_dir}/results'

            if args.save_subset:
                print(
                    f'Saving the results to {save_path}_subset')

                np.savez(f'{save_path}_subset',
                         train_loss=train_loss, test_acc=test_acc, train_acc=train_acc, test_loss=test_loss,
                         data_time=data_time, train_time=train_time, grd_time=grd_time, sim_time=sim_time,
                         best_g=best_gs, best_b=best_bs, not_selected=not_selected, times_selected=times_selected,
                         subset=selected_ndx, weights=selected_wgt)
            else:
                print(
                    f'Saving the results to {save_path}')

                np.savez(save_path,
                         train_loss=train_loss, test_acc=test_acc, train_acc=train_acc, test_loss=test_loss,
                         data_time=data_time, train_time=train_time, grd_time=grd_time, sim_time=sim_time,
                         best_g=best_gs, best_b=best_bs, not_selected=not_selected,
                         times_selected=times_selected)

    print(np.max(test_acc, 1), np.mean(np.max(test_acc, 1)),
          np.min(not_selected, 1), np.mean(np.min(not_selected, 1)))




def train(train_loader, model, epoch, criterion, optimizer, weight=None):
    """
        Run one train epoch
    """
    if weight is None:
        weight = torch.ones(len(train_loader.dataset)).cuda()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, idx) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        # print(weight[idx.long()])
        # loss = loss * weight[idx.long()]
        loss = loss.mean()  # (Note)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    return top1.avg, losses.avg, data_time.sum, batch_time.sum


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #               i, len(val_loader), batch_time=batch_time, loss=losses,
            #               top1=top1))

    print(' * Prec@1 {top1.avg:.3f}' .format(top1=top1))

    return top1.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def predictions(args, loader, model):
    """
    Get predictions
    """
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    preds = torch.zeros(args.train_num, args.class_num).cuda()
    labels = torch.zeros(args.train_num, dtype=torch.int)
    end = time.time()
    with torch.no_grad():
        for i, (input, target, idx) in enumerate(loader):
            input_var = input.cuda()

            if args.half:
                input_var = input_var.half()

            preds[idx, :] = nn.Softmax(dim=1)(model(input_var))
            labels[idx] = target.int()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Predict: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'
                      .format(i, len(loader), batch_time=batch_time))

    return preds.cpu().data.numpy(), labels.cpu().data.numpy()

def quantization_predictions(args, loader, model):
    model.to('cpu')
    model.eval()
    preds = np.zeros((args.train_num, args.class_num))
    labels = np.zeros(args.train_num)
    labels=labels.astype('int32')
    for i, (input, target, idx) in enumerate(loader):
        preds[idx, :] = nn.Softmax(dim=1)(model(input))
        labels[idx] = target.int()
    return preds, labels


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args, subset_size=args.subset_size, greedy=args.greedy)


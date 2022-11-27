import os
import subprocess
import time
import gc
import numpy as np
from lazy_greedy import FacilityLocation, lazy_greedy_heap
from sklearn.metrics import pairwise_distances
from submodlib.functions.facilityLocation import FacilityLocationFunction
import torchvision.transforms as transforms
import torchvision

SEED = 100
EPS = 1E-8
PLOT_NAMES = ['lr', 'data_loss', 'epoch_loss', 'test_loss']  # 'cluster_compare', 'cosine_compare', 'euclidean_compare'


def load_dataset(dataset, dataset_dir):
    '''
    Args
    - dataset: str, one of ['cifar10', 'covtype'] or filename in `data/`
    - dataset_dir: str, path to `data` folder

    Returns
    - X: np.array, shape [N, d]
      - exception: shape [N, 32, 32, 3] for cifar10
    - y: np.array, shape [N]
    '''
    if dataset == 'cifar10':
        path = os.path.join(dataset_dir, 'cifar10', 'cifar10.npz')
        with np.load(path) as npz:
            X = npz['x']  # shape [60000, 32, 32, 3], type uint8
            y = npz['y']  # shape [60000], type uint8
        # convert to float in (0, 1), center at mean 0
        X = X.astype(np.float32) / 255
        # X -= np.mean(X, axis=0)
    elif dataset == 'cifar10_features':
        path = os.path.join(dataset_dir, 'cifar10', 'train_features.npz')
        with np.load(path) as npz:
            X = npz['features']  # shape [50000, 64], type float32
            y = npz['labels']  # shape [50000], type int64
    elif dataset == 'cifar10_grads':
        # labels
        path = os.path.join(dataset_dir, 'cifar10', 'train_features.npz')
        with np.load(path) as npz:
            y = npz['labels']  # shape [50000], type int64
        # feautres
        path = os.path.join('grad_features.npy')
        X = np.load(path)  # shape [50000, 1000], type float16

    else:
        num, dim, name = 0, 0, ''
        if dataset == 'covtype':
            num, dim = 581012, 54
            name = 'covtype.libsvm.binary.scale'
        elif dataset == 'ijcnn1.t' or dataset == 'ijcnn1.tr':
            num, dim = 49990 if 'tr' in dataset else 91701, 22
            name = dataset
        elif dataset == 'combined_scale' or dataset == 'combined_scale.t':
            num, dim = 19705 if '.t' in dataset else 78823, 100
            name = dataset

        X = np.zeros((num, dim), dtype=np.float32)
        y = np.zeros(num, dtype=np.int32)
        path = os.path.join(dataset_dir, name)

        with open(path, 'r') as f:
            for i, line in enumerate(f):
                y[i] = float(line.split()[0])
                for e in line.split()[1:]:
                    cur = e.split(':')
                    X[i][int(cur[0]) - 1] = float(cur[1])
                i += 1
        y = np.array(y, dtype=np.int32)
        if name in ['ijcnn1.t', 'ijcnn1.tr']:
            y[y == -1] = 0
        else:
            y = y - np.ones(len(y), dtype=np.int32)

    return X, y


def similarity(X, metric):
    '''Computes the similarity between each pair of examples in X.

    Args
    - X: np.array, shape [N, d]
    - metric: str, one of ['cosine', 'euclidean']

    Returns
    - S: np.array, shape [N, N]
    '''
    # print(f'Computing similarity for {metric}...', flush=True)
    start = time.time()
    dists = pairwise_distances(X, metric=metric, n_jobs=1)
    # dists = gdist(X, X, optimize_level=0, output='cpu')
    elapsed = time.time() - start

    if metric == 'cosine':
        S = 1 - dists
    elif metric == 'euclidean' or metric == 'l1':
        m = np.max(dists)
        S = m - dists
    else:
        raise ValueError(f'unknown metric: {metric}')

    return S, elapsed


def get_facility_location_submodular_order(S, B, c, smtk=0, no=0, stoch_greedy=0, weights=None):
    '''
    Args
    - S: np.array, shape [N, N], similarity matrix
    - B: int, number of points to select

    Returns
    - order: np.array, shape [B], order of points selected by facility location
    - sz: np.array, shape [B], type int64, size of cluster associated with each selected point
    '''
    # print('Computing facility location submodular order...')
    N = S.shape[0]
    no = smtk if no == 0 else no

    if smtk > 0:
        print(f'Calculating ordering with SMTK... part size: {len(S)}, B: {B}', flush=True)
        np.save(f'/tmp/{no}/{smtk}-{c}', S)
        if stoch_greedy > 0:
            p = subprocess.check_output(
                f'/tmp/{no}/smtk-master{smtk}/build/smraiz -sumsize {B} \
                 -stochastic-greedy -sg-epsilon {stoch_greedy} -flnpy /tmp/{no}/{smtk}-{c}.'
                f'npy -pnpv -porder -ptime'.split())
        else:
            p = subprocess.check_output(
                f'/tmp/{no}/smtk-master{smtk}/build/smraiz -sumsize {B} \
                             -flnpy /tmp/{no}/{smtk}-{c}.npy -pnpv -porder -ptime'.split())
        s = p.decode("utf-8")
        str, end = ['([', ',])']
        order = s[s.find(str) + len(str):s.rfind(end)].split(',')
        greedy_time = float(s[s.find('CPU') + 4 : s.find('s (User')])
        str = 'f(Solution) = '
        F_val = float(s[s.find(str) + len(str) : s.find('Summary Solution') - 1])
    else:
        V = list(range(N))
        start = time.time()
        F = FacilityLocation(S, V)
        order, _ = lazy_greedy_heap(F, V, B)
        greedy_time = time.time() - start
        F_val = 0

        order = np.asarray(order, dtype=np.int64)
        sz = np.zeros(B, dtype=np.float64)
        for i in range(N):
            if weights is None:
                sz[np.argmax(S[i, order])] += 1
            else:
                sz[np.argmax(S[i, order])] += weights[i]
    # print('time (sec) for computing facility location:', greedy_time, flush=True)
    collected = gc.collect()
    return order, sz, greedy_time, F_val


# def faciliy_location_order(c, X, y, metric, num_per_class, smtk, no, stoch_greedy, weights=None):
#     class_indices = np.where(y == c)[0]
#     # print(class_indices)
#     # print(X)
#     print(f'Selecting from {len(class_indices)} examples in class {c}')
#     S, S_time = similarity(X[class_indices], metric=metric)
#     order, cluster_sz, greedy_time, F_val = get_facility_location_submodular_order(
#         S, num_per_class, c, smtk, no, stoch_greedy, weights)
#     return class_indices[order], cluster_sz, greedy_time, S_time

def faciliy_location_order(c, X, y, metric, num_per_class, smtk, no, stoch_greedy, weights=None, mode='dense', num_n=128):
    class_indices = np.where(y == c)[0]
    X = X[class_indices]
    N = X.shape[0]

    if mode == 'dense':
        num_n = None

    start = time.time()
    obj = FacilityLocationFunction(n=len(X), mode=mode, data=X, metric=metric, num_neighbors=num_n)
    S_time = time.time() - start

    start = time.time()
    greedyList = obj.maximize(
        budget=num_per_class,
        optimizer="LazyGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
    )
    order = list(map(lambda x: x[0], greedyList))
    sz = list(map(lambda x: x[1], greedyList))
    greedy_time = time.time() - start

    S = obj.sijs
    order = np.asarray(order, dtype=np.int64)
    sz = np.zeros(num_per_class, dtype=np.float64)

    for i in range(N):
        if np.max(S[i, order]) <= 0:
            continue
        if weights is None:
            sz[np.argmax(S[i, order])] += 1
        else:
            sz[np.argmax(S[i, order])] += weights[i]
    sz[np.where(sz==0)] = 1

    return class_indices[order], sz, greedy_time, S_time

def get_orders_and_weights(B, X, metric, smtk, no=0, stoch_greedy=0, y=None, weights=None, equal_num=False, outdir='.'):
    '''
    Ags
    - X: np.array, shape [N, d]
    - B: int, number of points to select
    - metric: str, one of ['cosine', 'euclidean'], for similarity
    - y: np.array, shape [N], integer class labels for C classes
      - if given, chooses B / C points per class, B must be divisible by C
    - outdir: str, path to output directory, must already exist

    Returns
    - order_mg/_sz: np.array, shape [B], type int64
      - *_mg: order points by their marginal gain in FL objective (largest gain first)
      - *_sz: order points by their cluster size (largest size first)
    - weights_mg/_sz: np.array, shape [B], type float32, sums to 1
    '''
    N = X.shape[0]
    if y is None:
        y = np.zeros(N, dtype=np.int32)  # assign every point to the same class
    classes = np.unique(y)
    C = len(classes)  # number of classes
    # assert np.array_equal(classes, np.arange(C))
    # assert B % C == 0

    if equal_num:
        class_nums = [sum(y == c) for c in classes]
        num_per_class = int(np.ceil(B / C)) * np.ones(len(classes), dtype=np.int32)
        minority = class_nums < np.ceil(B / C)
        if sum(minority) > 0:
            extra = sum([max(0, np.ceil(B / C) - class_nums[c]) for c in classes])
            for c in classes[~minority]:
                num_per_class[c] += int(np.ceil(extra / sum(minority)))
    else:
        num_per_class = np.int32(np.ceil(np.divide([sum(y == i) for i in classes], N) * B))
        print('not equal_num')

    # print(f'Greedy: selecting {num_per_class} elements')

    # order_mg_all = np.zeros([C, num_per_class], dtype=np.int64)
    # cluster_sizes_all = np.zeros([C, num_per_class], dtype=np.float32)
    # greedy_time_all = np.zeros([C, num_per_class], dtype=np.int64)
    # similarity_time_all = np.zeros([C, num_per_class], dtype=np.int64)

    # pool = ThreadPool(C)
    # order_mg_all, cluster_sizes_all, greedy_times, similarity_times = zip(*pool.map(
    #     lambda c: faciliy_location_order(c, X, y, metric, num_per_class[c], smtk, stoch_greedy, weights), classes))
    # pool.terminate()
    order_mg_all, cluster_sizes_all, greedy_times, similarity_times = zip(*map(
        lambda c: faciliy_location_order(c, X, y, metric, num_per_class[c], smtk, no, stoch_greedy, weights), classes))

    order_mg, weights_mg = [], []
    if equal_num:
        props = np.rint([len(order_mg_all[i]) for i in range(len(order_mg_all))])
    else:
        # merging imbalanced classes
        class_ratios = np.divide([np.sum(y == i) for i in classes], N)
        props = np.rint(class_ratios / np.min(class_ratios))
        print(f'Selecting with ratios {np.array(class_ratios)}')
        print(f'Class proportions {np.array(props)}')

    order_mg_all = np.array(order_mg_all)
    cluster_sizes_all = np.array(cluster_sizes_all)
    for i in range(int(np.rint(np.max([len(order_mg_all[c]) / props[c] for c in classes])))):
        for c in classes:
            ndx = slice(i * int(props[c]), int(min(len(order_mg_all[c]), (i + 1) * props[c])))
            order_mg = np.append(order_mg, order_mg_all[c][ndx])
            weights_mg = np.append(weights_mg, cluster_sizes_all[c][ndx])
    order_mg = np.array(order_mg, dtype=np.int32)

    # class_ratios = np.divide([np.sum(y == i) for i in classes], N)
    # weights_mg[y[order_mg] == np.argmax(class_ratios)] /= (np.max(class_ratios) / np.min(class_ratios))

    weights_mg = np.array(weights_mg, dtype=np.float32)
    ordering_time = np.max(greedy_times)
    similarity_time = np.max(similarity_times)

    # for c in classes:
    #     class_indices = np.where(y == c)[0]
    #     S, similarity_time_all[c] = similarity(X[class_indices], metric=metric)
    #     order, cluster_sz, greedy_time_all[c], F_val = get_facility_location_submodular_order(S, num_per_class, c, smtk)
    #     order_mg_all[c] = class_indices[order]
    #     cluster_sizes_all[c] = cluster_sz
    #     save_cluster_sizes(cluster_sizes_all[c], metric=f'{metric}_class{c}', outdir=outdir)
    # cluster_sizes_all /= N

    # choose 1st from each class, then 2nd from each class, etc.
    # i.e. column-major order
    # order_mg_all = np.array(order_mg_all)
    # cluster_sizes_all = np.array(cluster_sizes_all, dtype=np.float32) / N
    # order_mg = order_mg_all.flatten(order='F')
    # weights_mg = cluster_sizes_all.flatten(order='F')

    # sort by descending cluster size within each class
    # cluster_order = np.argsort(-cluster_sizes_all, axis=1)
    # rows_selector = np.arange(C)[:, np.newaxis]
    order_sz = []  # order_mg_all[rows_selector, cluster_order].flatten(order='F')
    weights_sz = [] # cluster_sizes_all[rows_selector, cluster_order].flatten(order='F')
    vals = order_mg, weights_mg, order_sz, weights_sz, ordering_time, similarity_time
    return vals

def get_dataset(args, train=True, train_transform=True):
    if args.dataset in ['cifar10', 'cifar100', 'mnist', 'svhn']:
        if args.dataset == 'cifar10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        elif args.dataset == 'cifar100':
            mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        elif args.dataset == 'svhn':
            mean = (0.4376821, 0.4437697, 0.47280442)
            std = (0.19803012, 0.20101562, 0.19703614)
        elif args.datast == 'mnist':
            mean = (0.1307,)
            std = (0.3081,)
        else:
            raise NotImplementedError

        if train and train_transform:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        if args.dataset == 'svhn':
            if train:
                dataset = torchvision.datasets.SVHN(
                    root=args.data_dir, split='train', 
                    transform=transform, download=True)
            else:
                dataset = torchvision.datasets.SVHN(
                    root=args.data_dir, split='test', 
                    transform=transform, download=True)
            dataset.targets = dataset.labels
        else:
            dataset = torchvision.datasets.__dict__[args.dataset.upper()](
                root=args.data_dir, train=train, 
                transform=transform, download=True)
    elif args.dataset == 'cinic10':
        mean = [0.47889522, 0.47227842, 0.43047404]
        std = [0.24205776, 0.23828046, 0.25874835]
        if train:
            path = args.data_dir + '/cinic-10/train'
        else:
            path = args.data_dir + '/cinic-10/test'
        if train and train_transform:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        dataset = torchvision.datasets.ImageFolder(path,
                transform=transform)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        if args.dataset == 'imagenet':
            import torch
            from datasets import load_dataset

            class HFWrapper(torch.utils.data.Dataset):
                def __init__(self, dataset, transform=None):
                    self.dataset = dataset
                    self.transform = transform
                    self.targets = dataset['label']

                def __getitem__(self, index):
                    batch = self.dataset[index]
                    data, target = batch['image'], batch['label']   
                    data = data.convert("RGB")
                    
                    if self.transform is not None:
                        data = self.transform(data)
                    return data, target

                def __len__(self):
                    return len(self.dataset)
            if train:
                dataset = load_dataset("imagenet-1k", use_auth_token=True, cache_dir=args.data_dir, split="train")
            else:
                dataset = load_dataset("imagenet-1k", use_auth_token=True, cache_dir=args.data_dir, split="test")
            dataset = HFWrapper(dataset, transform)
            
        elif args.dataset == 'tinyimagenet':
            if train:
                data_dir = os.path.join(args.data_dir, 'tiny-imagenet-200/train')
            else:
                data_dir = os.path.join(args.data_dir, 'tiny-imagenet-200/val')

            dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)

    return dataset

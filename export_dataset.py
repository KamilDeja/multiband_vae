import pickle
import sys
import argparse
import random
import torch

from torch.utils.data import DataLoader

import continual_benchmark.dataloaders.base
import continual_benchmark.dataloaders as dataloaders
from continual_benchmark.dataloaders.datasetGen import data_split

from visualise import *


def run(args):
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    train_dataset, val_dataset = dataloaders.base.__dict__[args.dataset](args.dataroot, args.skip_normalization,
                                                                         args.train_aug)

    if args.dataset.lower() == "celeba":
        n_classes = 10
    else:
        n_classes = train_dataset.number_classes

    train_dataset_splits, val_dataset_splits, task_output_space = data_split(dataset=train_dataset,
                                                                             dataset_name=args.dataset.lower(),
                                                                             num_batches=args.num_batches,
                                                                             num_classes=n_classes,
                                                                             random_split=args.random_split,
                                                                             random_mini_shuffle=args.random_shuffle,
                                                                             limit_data=args.limit_data,
                                                                             dirichlet_split_alpha=args.dirichlet,
                                                                             reverse=args.reverse)


    train_data = []
    labels = []
    save_path = f"{args.dataroot}/exported/{args.exported_name}_{len(train_dataset_splits)}_batches_random_{args.random_split}_train"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for idx in range(args.num_batches):
        train_dataset = train_dataset_splits[idx]
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        batch = next(iter(train_loader))
        batch_data, batch_labels = batch[0], batch[1]
        train_data.append(batch_data.numpy())
        labels.append(batch_labels.numpy())
        np.save(f"{save_path}/data_{idx}", batch_data.numpy())
        np.save(f"{save_path}/labels_{idx}", batch_labels.numpy())


def get_args(argv):
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--rpath', type=str, default='results/', help='Directory to save results')
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--seed', type=int, required=False,
                        help="Random seed. If defined all random operations will be reproducible")

    # Data
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='MNIST', help="MNIST(default)|CelebA")
    parser.add_argument('--exported_name', type=str, help="name of the exported dataset")
    parser.add_argument('--n_permutation', type=int, default=0, help="Enable permuted tests when >0")
    parser.add_argument('--num_batches', type=int, default=5)
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits")
    parser.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                        help="Randomize the order of splits")
    parser.add_argument('--random_split', dest='random_split', default=False, action='store_true',
                        help="Randomize data in splits")
    parser.add_argument('--random_shuffle', dest='random_shuffle', default=False, action='store_true',
                        help="Move part of data to next batch")
    parser.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true',
                        help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,6 ...] -> [0,1,2 ...]")
    parser.add_argument('--skip_normalization', action='store_true', help='Loads dataset without normalization')
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--workers', type=int, default=0, help="#Thread for dataloader")
    parser.add_argument('--limit_data', type=float, default=None,
                        help="limit_data to given %")
    parser.add_argument('--dirichlet', default=None, type=float,
                        help="Alpha parameter for dirichlet data split")
    parser.add_argument('--reverse', dest='reverse', default=False, action='store_true',
                        help="Reverse the ordering of batches")

    args = parser.parse_args(argv)

    return args


if __name__ == '__main__':
    args = get_args(sys.argv[1:])

    torch.cuda.set_device(args.gpuid[0])
    device = torch.device("cuda")

    if args.seed:
        print("Using manual seed = {}".format(args.seed))

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("WARNING: Not using manual seed - your experiments will not be reproducible")

    run(args)

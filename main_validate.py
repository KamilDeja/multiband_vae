import os
import sys
import argparse
import torch
import numpy as np
from collections import OrderedDict

import continual_benchmark.dataloaders.base
from continual_benchmark import dataloaders
from continual_benchmark.dataloaders.datasetGen import data_split
import torch.utils.data as data

from vae_experiments.validation import Validator
from visualise import plot_final_results
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def plot_examples(experiment_name, example, n_tasks):
    fig = plt.figure(figsize=(10., 10.))
    n_img = len(example) // (n_tasks + 1)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(n_tasks + 1, n_img),
                     axes_pad=0.5,
                     )

    for ax, im in zip(grid, example):  # sampled_classes):
        im = np.swapaxes(im, 0, 2)
        im = np.swapaxes(im, 0, 1)
        ax.imshow(im.squeeze())
        # ax.set_title('Task id: {}'.format(int(target)))

    plt.savefig("results/" + experiment_name + "/generations_task_" + str(n_tasks))
    plt.close()


def evaluate_directory(args, device):
    train_dataset, val_dataset = dataloaders.base.__dict__[args.dataset](args.dataroot, args.skip_normalization,
                                                                         False)
    if args.dataset.lower() == "celeba":
        n_classes = 10
    else:
        n_classes = train_dataset.number_classes
    n_batches = args.num_batches
    train_dataset_splits, val_dataset_splits, task_output_space = data_split(dataset=train_dataset,
                                                                             dataset_name=args.dataset.lower(),
                                                                             num_batches=n_batches,
                                                                             num_classes=n_classes,
                                                                             random_split=args.random_split,
                                                                             random_mini_shuffle=args.random_shuffle,
                                                                             limit_data=args.limit_data,
                                                                             dirichlet_split_alpha=args.dirichlet)
    val_loaders = []
    for task_name in range(n_batches):
        val_data = val_dataset_splits[
            task_name] if args.score_on_val else train_dataset_splits[task_name]
        val_loader = data.DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False,
                                     num_workers=args.workers)
        val_loaders.append(val_loader)

    validator = Validator(n_classes=n_classes, device=device, dataset=args.dataset,
                          stats_file_name=
                          f"compare_files_{args.dataset}_{args.directory}",
                          score_model_device=device, dataloaders=val_loaders)

    fid_table = OrderedDict()
    precision_table = OrderedDict()
    recall_table = OrderedDict()
    for task_id in range(n_batches):
        fid_table[task_id] = OrderedDict()
        precision_table[task_id] = OrderedDict()
        recall_table[task_id] = OrderedDict()
        to_plot = []
        print(f"Validation for task: {task_id}")
        for j in range(task_id + 1):
            val_name = j
            # print('validation split name:', val_name)
            examples = np.load(f"{args.directory}/generations_{task_id + 1}_{j + 1}.npy")
            if args.dataset.lower() in ["mnist", "fashionmnist", "omniglot", "doublemnist"]:
                examples = examples.reshape([-1, 1, 28, 28])
                # print(examples.size())
                # if args.dataset.lower() in ["fashionmnist", "doublemnist"]:
                #     examples = examples.repeat([1, 3, 1, 1])
            print(examples.shape)
            to_plot.append(examples[:5])
            fid_result, precision, recall = validator.compute_fid_from_examples(args, examples, j)  # task_id != 0)
            fid_table[j][task_id] = fid_result
            precision_table[j][task_id] = precision
            recall_table[j][task_id] = recall
            print(f"FID task {j}: {fid_result}")
        if args.dataset.lower() in ["mnist", "fashionmnist", "omniglot", "doublemnist"]:
            to_plot = np.concatenate(to_plot).reshape([-1, 1, 28, 28])
        elif args.dataset.lower() == "celeba":
            to_plot = np.concatenate(to_plot).reshape(-1, 3, 64, 64)
        else:
            raise NotImplementedError
        plot_examples(args.experiment_name, to_plot, task_id)

    return fid_table, precision_table, recall_table


def get_args(argv):
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--experiment_name', type=str, default='default_run', help='Name of current experiment')
    parser.add_argument('--directory', type=str, required=True, help='Directory with generations')
    parser.add_argument('--rpath', type=str, default='results/', help='Directory to save results')
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--seed', type=int, required=False, default=11,
                        help="Random seed. If defined all random operations will be reproducible")
    parser.add_argument('--num_batches', type=int, default=5)

    parser.add_argument('--val_batch_size', type=int, default=250)
    parser.add_argument('--dataset', type=str, default='MNIST', help="MNIST(default)|CelebA")
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--skip_normalization', action='store_true', help='Loads dataset without normalization')
    parser.add_argument('--score_on_val', action='store_true', required=False, default=False,
                        help="Compute FID on validation dataset instead of validation dataset")
    parser.add_argument('--random_split', dest='random_split', default=False, action='store_true',
                        help="Randomize data in splits")
    parser.add_argument('--random_shuffle', dest='random_shuffle', default=False, action='store_true',
                        help="Move part of data to next batch")
    parser.add_argument('--workers', type=int, default=0, help="#Thread for dataloader")
    parser.add_argument('--limit_data', type=float, default=None,
                        help="limit_data to given %")
    parser.add_argument('--dirichlet', default=None, type=float,
                        help="Alpha parameter for dirichlet data split")

    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    if args.seed:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.cuda.set_device(args.gpuid[0])
    device = torch.device("cuda")
    os.makedirs(f"{args.rpath}{args.experiment_name}", exist_ok=True)
    with open(f"{args.rpath}{args.experiment_name}/args.txt", "w") as text_file:
        text_file.write(str(args))
    acc_val, precision, recall = evaluate_directory(args, device)
    acc_val_dict, acc_test, precision_table, recall_table = {}, {}, {}, {}
    acc_val_dict[0] = acc_val
    precision_table[0], recall_table[0] = precision, recall
    np.save(f"{args.rpath}{args.experiment_name}/fid.npy", acc_val_dict)
    np.save(f"{args.rpath}{args.experiment_name}/precision.npy", precision_table)
    np.save(f"{args.rpath}{args.experiment_name}/recall.npy", recall_table)
    plot_final_results([args.experiment_name], type="fid")

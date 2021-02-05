import sys
import argparse
import copy
import random
import torch
import torch.utils.data as data
from random import shuffle
from collections import OrderedDict

import continual_benchmark.dataloaders.base
import continual_benchmark.agents as agents
import continual_benchmark.dataloaders as dataloaders
from continual_benchmark.dataloaders.datasetGen import SplitGen, PermutedGen, data_split
from vae_experiments import multiband_training, replay_training

from vae_experiments import training_functions
from vae_experiments import vae_utils
from vae_experiments.validation import Validator

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
    n_batches = n_classes // args.other_split_size
    train_dataset_splits, val_dataset_splits, task_output_space = data_split(dataset=train_dataset,
                                                                             dataset_name=args.dataset.lower(),
                                                                             num_batches=n_batches,
                                                                             num_classes=n_classes,
                                                                             random_split=args.random_split,
                                                                             random_mini_shuffle=args.random_shuffle)
    # else:
    #     # from vae_experiments import models_definition_mnist as models_definition
    #     train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
    #                                                                            first_split_sz=args.first_split_size,
    #                                                                            other_split_sz=args.other_split_size,
    #                                                                            rand_split=args.rand_split,
    #                                                                            remap_class=not args.no_class_remap,
    #                                                                            random_split=args.random_split)
    #

    if args.training_procedure == "replay":
        from vae_experiments import models_definition_replay as models_definition
    else:
        from vae_experiments import models_definition
    # Calculate constants

    labels_tasks = {}
    for task_name, task in train_dataset_splits.items():
        labels_tasks[int(task_name)] = task.dataset.class_list

    n_tasks = len(labels_tasks)

    # n_channels = val_dataset.dataset[0][0].size()[0]
    # in_size = val_dataset.dataset[0][0].size()[1]

    # Decide split ordering
    task_names = sorted(list(task_output_space.keys()), key=int)
    print('Task order:', task_names)
    if args.rand_split_order:
        shuffle(task_names)
        print('Shuffled task order:', task_names)
    fid_table = OrderedDict()
    test_fid_table = OrderedDict()

    # Prepare VAE
    if args.training_procedure == "multiband":
        local_vae = models_definition.VAE(latent_size=args.gen_latent_size, d=args.gen_d, p_coding=args.gen_p_coding,
                                          n_dim_coding=args.gen_n_dim_coding, cond_p_coding=args.gen_cond_p_coding,
                                          cond_n_dim_coding=args.gen_cond_n_dim_coding, cond_dim=n_classes,
                                          device=device, standard_embeddings=args.standard_embeddings,
                                          trainable_embeddings=args.trainable_embeddings,
                                          in_size=train_dataset[0][0].size()[1]).to(device)
    elif args.training_procedure == "replay":
        local_vae = models_definition.VAE(latent_size=args.gen_latent_size, d=args.gen_d, p_coding=args.gen_p_coding,
                                          n_dim_coding=args.gen_n_dim_coding, cond_p_coding=args.gen_cond_p_coding,
                                          cond_n_dim_coding=args.gen_cond_n_dim_coding, cond_dim=n_classes,
                                          device=device, in_size=train_dataset[0][0].size()[1]).to(device)

    print(local_vae)
    class_table = torch.zeros(n_tasks, n_classes, dtype=torch.long)

    train_loaders = []
    val_loaders = []
    for task_name in range(n_tasks):
        train_dataset_loader = data.DataLoader(dataset=train_dataset_splits[task_name],
                                               batch_size=args.gen_batch_size, shuffle=True,
                                               drop_last=False)
        train_loaders.append(train_dataset_loader)
        val_data = val_dataset_splits[
            task_name] if args.score_on_val else train_dataset_splits[task_name]
        val_loader = data.DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False,
                                     num_workers=args.workers)
        val_loaders.append(val_loader)

    validator = Validator(n_classes=n_classes, device=device, dataset=args.dataset,
                          stats_file_name=
                          f"seed_{args.seed}_f_split_{args.first_split_size}_split_{args.other_split_size}_val_{args.score_on_val}_{args.random_split}",
                          score_model_device=device, dataloaders=val_loaders)
    curr_global_decoder = None
    for task_id in range(len(task_names)):
        print("######### Task number {} #########".format(task_id))
        task_name = task_names[task_id]

        # VAE
        print("Train local VAE model")
        train_dataset_loader = train_loaders[task_id]

        if args.training_procedure == "multiband":
            curr_global_decoder = multiband_training.train_multiband(args=args, models_definition=models_definition,
                                                                     local_vae=local_vae,
                                                                     curr_global_decoder=curr_global_decoder,
                                                                     task_id=task_id,
                                                                     train_dataset_loader=train_dataset_loader,
                                                                     class_table=class_table, n_classes=n_classes,
                                                                     device=device)
        elif args.training_procedure == "replay":
            curr_global_decoder, tmp_table = replay_training.train_with_replay(args=args, local_vae=local_vae,
                                                                               task_loader=train_dataset_loader,
                                                                               task_id=task_id, class_table=class_table)
            class_table[task_id] = tmp_table
        else:
            print("Wrong training procedure")
            return None

        # Plotting results for already learned tasks
        if not args.gen_load_pretrained_models:
            vae_utils.plot_results(args.experiment_name, curr_global_decoder, class_table, task_id, same_z=True)
            vae_utils.plot_results(args.experiment_name, local_vae.decoder, class_table, task_id,
                                   translate_noise=task_id == 0, suffix="_local_vae", same_z=True)
            torch.save(curr_global_decoder.state_dict(), f"results/{args.experiment_name}/model{task_id}_curr_decoder")
            torch.save(local_vae.state_dict(), f"results/{args.experiment_name}/model{task_id}_local_vae")

        # local_vae.decoder = curr_global_decoder

        # Classifier validation
        fid_table[task_name] = OrderedDict()
        if args.skip_validation:
            for j in range(task_id + 1):
                fid_table[j][task_name] = -1
        else:
            for j in range(task_id + 1):
                val_name = task_names[j]
                print('validation split name:', val_name)
                fid_result = validator.compute_fid(curr_global_decoder=curr_global_decoder, class_table=class_table,
                                                   task_id=j, translate_noise=task_id != 0)
                fid_table[j][task_name] = fid_result
                print(f"FID task {j}: {fid_result}")
    return fid_table, task_names, test_fid_table


def get_args(argv):
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--experiment_name', type=str, default='default_run', help='Name of current experiment')
    parser.add_argument('--rpath', type=str, default='results/', help='Directory to save results')
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--seed', type=int, required=False,
                        help="Random seed. If defined all random operations will be reproducible")
    parser.add_argument('--score_on_val', action='store_true', required=False, default=False,
                        help="Compute FID on validation dataset instead of validation dataset")
    parser.add_argument('--val_batch_size', type=int, default=250)
    parser.add_argument('--skip_validation', default=False, action='store_true')
    parser.add_argument('--training_procedure', type=str, default='multiband',
                        help='Training procedure multiband|replay')

    # Data
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='MNIST', help="MNIST(default)|CelebA")
    parser.add_argument('--n_permutation', type=int, default=0, help="Enable permuted tests when >0")
    parser.add_argument('--first_split_size', type=int, default=2)
    parser.add_argument('--other_split_size', type=int, default=2)
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

    # Generative network - multiband vae
    parser.add_argument('--gen_batch_size', type=int, default=50)
    parser.add_argument('--gen_n_dim_coding', type=int, default=4,
                        help="Number of bits used to code task id in binary autoencoder")
    parser.add_argument('--gen_p_coding', type=int, default=9,
                        help="Prime number used to calculated codes in binary autoencoder")
    parser.add_argument('--gen_cond_n_dim_coding', type=int, default=4,
                        help="Number of bits used to code task id in binary autoencoder")
    parser.add_argument('--gen_cond_p_coding', type=int, default=9,
                        help="Prime number used to calculated codes in binary autoencoder")
    parser.add_argument('--gen_latent_size', type=int, default=10, help="Latent size in binary autoencoder")
    parser.add_argument('--gen_d', type=int, default=8, help="Size of binary autoencoder")
    parser.add_argument('--gen_ae_epochs', type=int, default=20,
                        help="Number of epochs to train local variational autoencoder")
    parser.add_argument('--global_dec_epochs', type=int, default=20, help="Number of epochs to train global decoder")
    parser.add_argument('--gen_load_pretrained_models', default=False, help="Load pretrained generative models")
    parser.add_argument('--gen_pretrained_models_dir', type=str, default="results/pretrained_models",
                        help="Directory of pretrained generative models")
    parser.add_argument('--standard_embeddings', dest='standard_embeddings', default=False, action='store_true',
                        help="Train multiband with standard embeddings instead of matrix")
    parser.add_argument('--trainable_embeddings', dest='trainable_embeddings', default=False, action='store_true',
                        help="Train multiband with trainable embeddings instead of matrix")

    args = parser.parse_args(argv)

    if args.trainable_embeddings:
        args.standard_embeddings = True

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

    acc_val, acc_test = {}, {}
    os.makedirs(f"{args.rpath}{args.experiment_name}", exist_ok=True)
    with open(f"{args.rpath}{args.experiment_name}/args.txt", "w") as text_file:
        text_file.write(str(args))
    for r in range(args.repeat):
        acc_val[r], _, acc_test[r] = run(args)
    np.save(f"{args.rpath}{args.experiment_name}/acc_val.npy", acc_val)
    np.save(f"{args.rpath}{args.experiment_name}/acc_test.npy", acc_test)
    plot_final_results([args.experiment_name])

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
from continual_benchmark.dataloaders.datasetGen import SplitGen, PermutedGen

from vae_experiments import models_definition
from vae_experiments import training_functions
from vae_experiments import vae_utils

from visualise import *


exp_values = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
parts = len(exp_values)


def run(args):
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    train_dataset, val_dataset = dataloaders.base.__dict__[args.dataset](args.dataroot, args.skip_normalization,
                                                                         args.train_aug)
    if args.n_permutation > 0:
        train_dataset_splits, val_dataset_splits, task_output_space = PermutedGen(train_dataset, val_dataset,
                                                                                  args.n_permutation,
                                                                                  remap_class=not args.no_class_remap)
    else:
        train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                               first_split_sz=args.first_split_size,
                                                                               other_split_sz=args.other_split_size,
                                                                               rand_split=args.rand_split,
                                                                               remap_class=not args.no_class_remap)

    # Calculate constants
    n_classes = train_dataset.number_classes

    labels_tasks = {}
    for task_name, task in train_dataset_splits.items():
        labels_tasks[int(task_name)] = task.dataset.class_list

    n_tasks = len(labels_tasks)

    n_channels = val_dataset.dataset[0][0].size()[0]
    in_size = val_dataset.dataset[0][0].size()[1]

    agent_config = {'lr': args.base_lr,
                    'momentum': args.base_momentum,
                    'nesterov': args.base_nesterov,
                    'weight_decay': args.base_weight_decay,
                    'base_schedule': args.base_schedule,
                    'base_model_type': args.base_model_type,
                    'base_model_name': args.base_model_name,
                    'base_model_weights': args.base_model_weights,
                    'out_dim': {'All': args.base_force_out_dim} if args.base_force_out_dim > 0 else task_output_space,
                    'optimizer': args.base_optimizer,
                    'base_print_freq': args.base_print_freq,
                    'score_generated_images_by_freezed_classifier': args.score_generated_images_by_freezed_classifier,
                    'gpuid': args.gpuid}

    agent = agents.default.NormalNN(agent_config, n_channels=n_channels, in_size=in_size, n_classes=n_classes,
                                    d=args.base_model_d, model_bn=args.base_model_bn,max_pool=args.base_max_pool, n_conv=args.base_n_conv,
                                    dropout_rate=args.base_dropout_rate)

    # Decide split ordering
    task_names = sorted(list(task_output_space.keys()), key=int)
    print('Task order:', task_names)
    if args.rand_split_order:
        shuffle(task_names)
        print('Shuffled task order:', task_names)
    acc_table = OrderedDict()
    test_acc_table = OrderedDict()

    # Prepare VAE
    local_vae = models_definition.VAE(latent_size=args.gen_latent_size, d=args.gen_d, p_coding=args.gen_p_coding,
                                      n_dim_coding=args.gen_n_dim_coding, device=device, n_channels=n_channels,
                                      in_size=in_size).to(device)

    print(local_vae)
    class_table = torch.zeros(n_tasks, n_classes, dtype=torch.long)
    global_classes_list = []
    global_n_codes = []

    for task_id in range(len(task_names)):
        print("######### Task number {} #########".format(task_id))
        task_name = task_names[task_id]

        # VAE
        print("Train local VAE model")
        n_codes = len(train_dataset_splits[task_name])
        train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset_splits[task_name],
                                                           batch_size=args.gen_batch_size, shuffle=True, drop_last=False)
        data_loader_stable = torch.utils.data.DataLoader(dataset=train_dataset_splits[task_name],
                                                         batch_size=args.gen_batch_size, shuffle=False, drop_last=False)
        data_loader_total = torch.utils.data.DataLoader(dataset=train_dataset_splits[task_name],
                                                        batch_size=n_codes, shuffle=False, drop_last=False)
        global_n_codes.append(n_codes)
        start_id = int(np.sum(global_n_codes[:task_id]))
        codes_range = range(start_id, start_id + n_codes)
        codes_rep = torch.Tensor()

        for exp_value in exp_values:
            codes = codes_range * np.array(
                exp_value ** np.floor(args.gen_latent_size // parts * np.log(2) / np.log(exp_value)),
                dtype=np.longlong) % 2 ** args.gen_latent_size // parts
            codes = torch.tensor(
                models_definition.unpackbits(np.array(codes, dtype=np.longlong), args.gen_latent_size // parts)).float()
            codes_rep = torch.cat([codes_rep, codes], 1)

        if args.gen_load_pretrained_models:
            codes_rep = (codes_rep.repeat([args.gen_batch_size, 1, 1]) * 2 - 1)
        else:
            codes_rep = (codes_rep.repeat([args.gen_batch_size, 1, 1]).to(device) * 2 - 1)

        if args.gen_load_pretrained_models:
            local_vae.load_state_dict(torch.load(args.gen_pretrained_models_dir + f'model{task_id}_local_vae'))
            global_classes_list = np.load(args.gen_pretrained_models_dir + f'model{task_id}_classes.npy')
        else:
            dataloader_with_codes = training_functions.train_local_generator(local_vae, train_dataset_loader,
                                                                             data_loader_stable,
                                                                             data_loader_total,
                                                                             global_classes_list, task_id, codes_rep,
                                                                             args.gen_batch_size,
                                                                             n_epochs_pre=args.gen_ae_pre_epochs,
                                                                             n_epochs=args.gen_ae_epochs)
        print("Done training local VAE model")
        del codes_rep

        if not task_id:
            # First task, initializing global decoder as local_vae's decoder
            curr_global_decoder = copy.deepcopy(local_vae.decoder)
        else:
            print("Train global VAE model")
            # Retraining global decoder with previous global decoder and local_vae
            if args.gen_load_pretrained_models:
                curr_global_decoder = models_definition.Decoder(local_vae.latent_size, args.gen_d*4,
                                                                p_coding=local_vae.p_coding,
                                                                n_dim_coding=local_vae.n_dim_coding,
                                                                device=local_vae.device,
                                                                n_channels=n_channels, in_size=in_size).to(
                    local_vae.device)
                curr_global_decoder.load_state_dict(
                    torch.load(args.gen_pretrained_models_dir + f'model{task_id}_curr_decoder'))
            else:
                curr_global_decoder = training_functions.train_global_decoder(curr_global_decoder, local_vae,
                                                                              dataloader_with_codes, task_id=task_id,
                                                                              codes_rep=None, total_n_codes=n_codes,
                                                                              global_n_codes=global_n_codes,
                                                                              global_classes_list=global_classes_list,
                                                                              d=args.gen_d,
                                                                              n_epochs=args.gen_ae_epochs,
                                                                              batch_size=args.gen_batch_size,
                                                                              n_channels=n_channels, in_size=in_size)
            torch.cuda.empty_cache()

        # Plotting results for already learned tasks
        if not args.gen_load_pretrained_models:
            vae_utils.plot_results(args.experiment_name, curr_global_decoder, task_id, n_codes, global_n_codes,
                                   global_classes_list)
            vae_utils.plot_results(args.experiment_name, local_vae.decoder, task_id, n_codes, global_n_codes,
                                   global_classes_list, 5, "_local_vae")
            torch.save(curr_global_decoder.state_dict(), f"results/{args.experiment_name}/model{task_id}_curr_decoder")
            torch.save(local_vae.state_dict(), f"results/{args.experiment_name}/model{task_id}_local_vae")
            torch.save(agent.model.state_dict(), f"results/{args.experiment_name}/model{task_id}_classifier")
            np.save(f"results/{args.experiment_name}/model{task_id}_classes", global_classes_list)

        # Classifier
        train_loader = data.DataLoader(train_dataset_splits[task_name],
                                       batch_size=args.base_batch_size,
                                       shuffle=True,
                                       num_workers=args.workers)

        val_loader = data.DataLoader(val_dataset_splits[task_name],
                                     batch_size=args.base_batch_size,
                                     shuffle=True,
                                     num_workers=args.workers)

        agent.learn_batch(train_loader, val_loader, curr_global_decoder, local_vae, class_table, global_classes_list,
                          task_id, n_codes, global_n_codes, args.new_task_data_processing)

        # Classifier validation
        acc_table[task_name] = OrderedDict()
        for j in range(task_id + 1):
            agent.active_neurons = torch.zeros((1, 4000))
            val_name = task_names[j]
            print('validation split name:', val_name)
            val_data = val_dataset_splits[val_name] if not args.base_eval_on_train_set else train_dataset_splits[val_name]
            val_loader = data.DataLoader(val_data,
                                         batch_size=args.base_batch_size,
                                         shuffle=True,
                                         num_workers=args.workers)
            acc_table[val_name][task_name] = agent.validation(val_loader)

    return acc_table, task_names, test_acc_table


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

    # Data
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='MNIST', help="MNIST(default)|FashionMNIST|CIFAR10|CIFAR100")
    parser.add_argument('--n_permutation', type=int, default=0, help="Enable permuted tests when >0")
    parser.add_argument('--first_split_size', type=int, default=2)
    parser.add_argument('--other_split_size', type=int, default=2)
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits")
    parser.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                        help="Randomize the order of splits")
    parser.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true',
                        help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,6 ...] -> [0,1,2 ...]")
    parser.add_argument('--skip_normalization', action='store_true', help='Loads dataset without normalization')
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--workers', type=int, default=0, help="#Thread for dataloader")

    # Learning options
    parser.add_argument('--new_task_data_processing', type=str,
                        choices=['original', 'original_through_vae', 'generated'],
                        default='original', help="Determines train data for base network.")
    parser.add_argument('--score_generated_images_by_freezed_classifier', default=True, action='store_true',
                        help="Score generated images by freezed classifier. If false - generator prompts the labels")

    # Base network - currently classfier
    parser.add_argument('--base_batch_size', type=int, default=100)
    parser.add_argument('--base_model_type', type=str, default='mlp',
                        help="The type (lenet|resnet|cifar_net) of backbone network")
    parser.add_argument('--base_model_name', type=str, default='MLP', help="The name of actual model for the backbone")
    parser.add_argument('--base_force_out_dim', type=int, default=2,
                        help="Set 0 to let the task decide the required output dimension")
    parser.add_argument('--base_schedule', nargs="+", type=int, default=[2],
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch")
    parser.add_argument('--base_print_freq', type=float, default=100, help="Print the log at every x iteration")
    parser.add_argument('--base_model_weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--base_eval_on_train_set', dest='base_eval_on_train_set', default=False, action='store_true',
                        help="Force the evaluation on train set")

    parser.add_argument('--base_model_d', type=int, default=64, help="Size of base network")
    parser.add_argument('--base_model_bn', default=True, help="Use batch norm in base network")
    parser.add_argument('--base_max_pool', default=False, help="Use max pooling in base network")
    parser.add_argument('--base_n_conv', type=int, default=3, help="Num of convs in base network")
    parser.add_argument('--base_dropout_rate', type=float, default=0.4, help="Dropout rate in base network")

    parser.add_argument('--base_optimizer', type=str, default='Adam',
                        help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--base_lr', type=float, default=0.01, help="Learning rate for base network")
    parser.add_argument('--base_nesterov', action='store_true', help='Whether to use nesterov momentum in base network')
    parser.add_argument('--base_momentum', type=float, default=0)
    parser.add_argument('--base_weight_decay', type=float, default=0)

    # Generative network - currently binary latent autoencoder
    parser.add_argument('--gen_batch_size', type=int, default=50)
    parser.add_argument('--gen_n_dim_coding', type=int, default=10,
                        help="Number of bits used to code task id in binary autoencoder")
    parser.add_argument('--gen_p_coding', type=int, default=307,
                        help="Prime number used to calculated codes in binary autoencoder")
    parser.add_argument('--gen_latent_size', type=int, default=200, help="Latent size in binary autoencoder")
    parser.add_argument('--gen_d', type=int, default=32, help="Size of binary autoencoder")
    parser.add_argument('--gen_ae_pre_epochs', type=int, default=20,
                        help="Number of epochs to train autoencoder before freezing the codes")
    parser.add_argument('--gen_ae_epochs', type=int, default=200, help="Number of epochs to train autoencoder")
    parser.add_argument('--gen_load_pretrained_models', default=False, help="Load pretrained generative models")
    parser.add_argument('--gen_pretrained_models_dir', type=str, default="results/pretrained_models",
                        help="Directory of pretrained generative models")

    args = parser.parse_args(argv)

    return args


if __name__ == '__main__':
    args = get_args(sys.argv[1:])

    torch.cuda.set_device(0)
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

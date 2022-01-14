import os

import torch
import numpy as np

from vae_experiments.fid import calculate_frechet_distance
from vae_experiments.prd import compute_prd_from_embedding, prd_to_max_f_beta_pair
from vae_experiments.vae_utils import generate_images
from scipy.stats import wasserstein_distance


class Validator:
    def __init__(self, n_classes, device, dataset, stats_file_name, dataloaders, score_model_device=None):
        self.n_classes = n_classes
        self.device = device
        self.dataset = dataset
        self.score_model_device = score_model_device
        self.dataloaders = dataloaders

        print("Preparing validator")
        if dataset in ["MNIST", "Omniglot"]:  # , "DoubleMNIST"]:
            if dataset in ["Omniglot"]:
                from vae_experiments.evaluation_models.lenet_Omniglot import Model
            # elif dataset == "DoubleMNIST":
            #     from vae_experiments.evaluation_models.lenet_DoubleMNIST import Model
            else:
                from vae_experiments.evaluation_models.lenet import Model
            net = Model()
            model_path = "vae_experiments/evaluation_models/lenet_" + dataset
            net.load_state_dict(torch.load(model_path))
            net.to(device)
            net.eval()
            self.dims = 128 if dataset in ["Omniglot", "DoubleMNIST"] else 84  # 128
            self.score_model_func = net.part_forward
        elif dataset.lower() in ["celeba", "doublemnist", "fashionmnist", "flowers", "cern"]:
            from vae_experiments.evaluation_models.inception import InceptionV3
            self.dims = 2048
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
            model = InceptionV3([block_idx])
            if score_model_device:
                model = model.to(score_model_device)
            model.eval()
            self.score_model_func = lambda batch: model(batch)[0]
        self.stats_file_name = f"{stats_file_name}_dims_{self.dims}"

    def calculate_results(self, curr_global_decoder, class_table, task_id, translate_noise=True, starting_point=None,
                          sample_tasks=False, dataset=None):
        curr_global_decoder.eval()
        class_table = class_table[:task_id + 1]
        test_loader = self.dataloaders[task_id]
        with torch.no_grad():
            distribution_orig = []
            distribution_gen = []
            if sample_tasks:
                task_samplers = []
                # This is the other way round, we first select class from the class table, and then we randomly sample
                # from tasks where this class was present

                for class_id in range(self.n_classes):
                    local_probs = class_table[:, class_id] * 1.0 / torch.sum(class_table[:, class_id])
                    task_samplers.append(torch.distributions.categorical.Categorical(probs=local_probs))

            precalculated_statistics = False
            os.makedirs(f"results/orig_stats/", exist_ok=True)
            stats_file_path = f"results/orig_stats/{self.dataset}_{self.stats_file_name}_{task_id}.npy"
            if os.path.exists(stats_file_path):
                print(f"Loading cached original data statistics from: {self.stats_file_name}")
                distribution_orig = np.load(stats_file_path)
                precalculated_statistics = True

            print("Calculating FID:")
            for idx, batch in enumerate(test_loader):
                x = batch[0].to(self.device)
                y = batch[1]
                z = torch.randn([len(y), curr_global_decoder.latent_size]).to(self.device)
                bin_z = torch.bernoulli(curr_global_decoder.ones_distribution[task_id].repeat([len(y), 1])).to(
                    self.device)
                # bin_z = torch.rand([len(y), curr_global_decoder.binary_latent_size]).to(self.device)
                bin_z = bin_z * 2 - 1
                y = y.sort()[0]
                labels, counts = torch.unique_consecutive(y, return_counts=True)
                if sample_tasks:
                    tasks_sampled = []
                    for i, n_occ in zip(labels, counts):
                        if task_samplers[i].probs.sum() > 0:  # n_occ > 0:
                            tasks_sampled.append(task_samplers[i].sample([n_occ]))
                        else:
                            tasks_sampled.append(task_samplers[class_table[task_id].argmax()].sample([n_occ]))

                    task_ids = torch.cat(tasks_sampled)
                else:
                    if starting_point != None:
                        task_ids = torch.zeros(len(y)) + starting_point
                    else:
                        task_ids = torch.zeros(len(y)) + task_id
                example = generate_images(curr_global_decoder, z, bin_z, task_ids, y, translate_noise=translate_noise)
                if not precalculated_statistics:
                    if dataset.lower() in ["fashionmnist", "doublemnist"]:
                        x = x.repeat([1, 3, 1, 1])
                    distribution_orig.append(self.score_model_func(x).cpu().detach().numpy())
                if dataset.lower() in ["fashionmnist", "doublemnist"]:
                    example = example.repeat([1, 3, 1, 1])
                distribution_gen.append(self.score_model_func(example))

            distribution_gen = torch.cat(distribution_gen).cpu().detach().numpy().reshape(-1, self.dims)
            # distribution_gen = np.array(np.concatenate(distribution_gen)).reshape(-1, self.dims)
            if not precalculated_statistics:
                distribution_orig = np.array(np.concatenate(distribution_orig)).reshape(-1, self.dims)
                np.save(stats_file_path, distribution_orig)

            precision, recall = compute_prd_from_embedding(
                eval_data=distribution_orig[np.random.choice(len(distribution_orig), len(distribution_gen), False)],
                ref_data=distribution_gen)
            precision, recall = prd_to_max_f_beta_pair(precision, recall)
            print(f"Precision:{precision},recall: {recall}")
            return calculate_frechet_distance(distribution_gen, distribution_orig), precision, recall

    def compute_results_from_examples(self, args, generations, task_id, join_tasks=False):
        distribution_orig = []
        precalculated_statistics = False
        stats_file_path = f"results/orig_stats/compare_files_{args.dataset}_{args.experiment_name}_{task_id}.npy"
        test_loader = self.dataloaders[task_id]

        if os.path.exists(stats_file_path) and not join_tasks:
            print(f"Loading cached original data statistics from: {self.stats_file_name}")
            distribution_orig = np.load(stats_file_path)
            precalculated_statistics = True
        print("Calculating FID:")
        if not precalculated_statistics:
            if join_tasks:
                for task in range(task_id + 1):
                    test_loader = self.dataloaders[task]
                    for idx, batch in enumerate(test_loader):
                        x = batch[0].to(self.device)
                        if args.dataset.lower() in ["fashionmnist", "doublemnist"]:
                            x = x.repeat([1, 3, 1, 1])
                        distribution_orig.append(self.score_model_func(x).cpu().detach().numpy())
            else:
                for idx, batch in enumerate(test_loader):
                    x = batch[0].to(self.device)
                    if args.dataset.lower() in ["fashionmnist", "doublemnist"]:
                        x = x.repeat([1, 3, 1, 1])
                    distribution_orig.append(self.score_model_func(x).cpu().detach().numpy())

        if args.dataset.lower() in ["mnist", "fashionmnist", "omniglot", "doublemnist"]:
            generations = generations.reshape(-1, 1, 28, 28)
        elif args.dataset.lower() in ["celeba", "flowers"]:
            generations = generations.reshape(-1, 3, 64, 64)
        generations = torch.from_numpy(generations).to(self.device)
        if args.dataset.lower() in ["fashionmnist", "doublemnist"]:
            generations = generations.repeat([1, 3, 1, 1])

        if not precalculated_statistics:
            distribution_orig = np.array(np.concatenate(distribution_orig)).reshape(-1, self.dims)
            if not join_tasks:
                os.makedirs("/".join(stats_file_path.split("/")[:-1]), exist_ok=True)
                np.save(stats_file_path, distribution_orig)

        distribution_gen = []
        batch_size = args.val_batch_size
        max_len = min(len(generations), len(distribution_orig))
        for idx in range(0, max_len, batch_size):
            start_point = idx
            end_point = min(max_len, idx + batch_size)
            distribution_gen.append(
                self.score_model_func(generations[start_point:end_point]).cpu().detach().numpy().reshape(-1, self.dims))
        distribution_gen = np.concatenate(distribution_gen)

        print(f"Orig:{len(distribution_orig)}, Gen:{len(distribution_gen)}")
        precision, recall = compute_prd_from_embedding(
            eval_data=distribution_orig,
            ref_data=distribution_gen[np.random.choice(len(distribution_gen), len(distribution_orig), True)]#TODO go back to FALSE
        )
        precision, recall = prd_to_max_f_beta_pair(precision, recall)
        print(f"Precision:{precision},recall: {recall}")

        return calculate_frechet_distance(
            distribution_gen[np.random.choice(len(distribution_gen), len(distribution_orig), True)], #TODO go back to FALSE
            distribution_orig), precision, recall


class CERN_Validator:
    def __init__(self, dataloaders, stats_file_name, device):
        self.dataloaders = dataloaders
        self.stats_file_name = stats_file_name
        self.device = device

    def sum_channels_parallel(self, data):
        coords = np.ogrid[0:data.shape[1], 0:data.shape[2]]
        half_x = data.shape[1] // 2
        half_y = data.shape[2] // 2

        checkerboard = (coords[0] + coords[1]) % 2 != 0
        checkerboard.reshape(-1, checkerboard.shape[0], checkerboard.shape[1])

        ch5 = (data * checkerboard).sum(axis=1).sum(axis=1)

        checkerboard = (coords[0] + coords[1]) % 2 == 0
        checkerboard = checkerboard.reshape(-1, checkerboard.shape[0], checkerboard.shape[1])

        mask = np.zeros((1, data.shape[1], data.shape[2]))
        mask[:, :half_x, :half_y] = checkerboard[:, :half_x, :half_y]
        ch1 = (data * mask).sum(axis=1).sum(axis=1)

        mask = np.zeros((1, data.shape[1], data.shape[2]))
        mask[:, :half_x, half_y:] = checkerboard[:, :half_x, half_y:]
        ch2 = (data * mask).sum(axis=1).sum(axis=1)

        mask = np.zeros((1, data.shape[1], data.shape[2]))
        mask[:, half_x:, :half_y] = checkerboard[:, half_x:, :half_y]
        ch3 = (data * mask).sum(axis=1).sum(axis=1)

        mask = np.zeros((1, data.shape[1], data.shape[2]))
        mask[:, half_x:, half_y:] = checkerboard[:, half_x:, half_y:]
        ch4 = (data * mask).sum(axis=1).sum(axis=1)

        # assert all(ch1+ch2+ch3+ch4+ch5 == data.sum(axis=1).sum(axis=1))==True

        return np.stack([ch1, ch2, ch3, ch4, ch5])

    def calculate_results(self, curr_global_decoder, class_table, task_id, translate_noise=True, starting_point=None,
                         sample_tasks=False, dataset=None):
        curr_global_decoder.eval()
        class_table = class_table[:task_id + 1]
        test_loader = self.dataloaders[task_id]
        with torch.no_grad():
            distribution_orig = []
            distribution_gen = []

            precalculated_statistics = False
            os.makedirs(f"results/orig_stats/", exist_ok=True)
            stats_file_path = f"results/orig_stats/CERN_{self.stats_file_name}_{task_id}.npy"
            if os.path.exists(stats_file_path):
                print(f"Loading cached original data statistics from: {self.stats_file_name}")
                distribution_orig = np.load(stats_file_path)
                precalculated_statistics = True

            print("Calculating CERN scores:")
            for idx, batch in enumerate(test_loader):
                x = batch[0].to(self.device)
                y = batch[1]
                z = torch.randn([len(y), curr_global_decoder.latent_size]).to(self.device)
                bin_z = torch.bernoulli(curr_global_decoder.ones_distribution[task_id].repeat([len(y), 1])).to(
                    self.device)
                # bin_z = torch.rand([len(y), curr_global_decoder.binary_latent_size]).to(self.device)
                bin_z = bin_z * 2 - 1
                y = y.sort()[0]
                # labels, counts = torch.unique_consecutive(y, return_counts=True)
                if starting_point != None:
                    task_ids = torch.zeros(len(y)) + starting_point
                else:
                    task_ids = torch.zeros(len(y)) + task_id
                example = generate_images(curr_global_decoder, z, bin_z, task_ids, y, translate_noise=translate_noise)
                if not precalculated_statistics:
                    # x = torch.exp(x)
                    distribution_orig.append(self.sum_channels_parallel(x.cpu().detach().numpy().squeeze(1)))
                # example = torch.exp(example)
                distribution_gen.append(self.sum_channels_parallel(example.cpu().detach().numpy().squeeze(1)))

            distribution_gen = np.hstack(distribution_gen)
            # distribution_gen = np.array(np.concatenate(distribution_gen)).reshape(-1, self.dims)
            if not precalculated_statistics:
                distribution_orig = np.hstack(distribution_orig)
                np.save(stats_file_path, distribution_orig)

            return wasserstein_distance(distribution_orig.reshape(-1), distribution_gen.reshape(-1)), 0, 0

    def compute_results_from_examples(self, args, generations, task_id, join_tasks=False):
        distribution_orig = []
        precalculated_statistics = False
        stats_file_path = f"results/orig_stats/compare_files_{args.dataset}_{args.experiment_name}_{task_id}.npy"
        test_loader = self.dataloaders[task_id]
        if os.path.exists(stats_file_path) and not join_tasks:
            print(f"Loading cached original data statistics from: {self.stats_file_name}")
            distribution_orig = np.load(stats_file_path)
            precalculated_statistics = True
        print("Calculating results:")
        if not precalculated_statistics:
            if join_tasks:
                for task in range(task_id + 1):
                    test_loader = self.dataloaders[task]
                    for idx, batch in enumerate(test_loader):
                        x = batch[0]
                        distribution_orig.append(self.sum_channels_parallel(x.numpy().squeeze(1)))
            else:
                for idx, batch in enumerate(test_loader):
                    x = batch[0]
                    distribution_orig.append(self.sum_channels_parallel(x.numpy().squeeze(1)))

        generations = generations.reshape(-1, 44, 44)
        # generations = torch.from_numpy(generations).to(self.device)

        if not precalculated_statistics:
            distribution_orig = np.hstack(distribution_orig)
            if not join_tasks:
                os.makedirs("/".join(stats_file_path.split("/")[:-1]), exist_ok=True)
                np.save(stats_file_path, distribution_orig)

        distribution_gen = []
        batch_size = args.val_batch_size
        # distribution_gen.append(self.sum_channels_parallel(example.cpu().detach().numpy().squeeze(1)))
        distribution_gen = self.sum_channels_parallel(generations)

        return wasserstein_distance(distribution_orig.reshape(-1), distribution_gen.reshape(-1)), 0, 0
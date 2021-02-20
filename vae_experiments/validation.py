import os

import torch
import numpy as np

from vae_experiments.fid import calculate_frechet_distance
from vae_experiments.prd import compute_prd_from_embedding, prd_to_max_f_beta_pair
from vae_experiments.vae_utils import generate_images


class Validator:
    def __init__(self, n_classes, device, dataset, stats_file_name, dataloaders, score_model_device=None):
        self.n_classes = n_classes
        self.device = device
        self.dataset = dataset
        self.score_model_device = score_model_device
        self.dataloaders = dataloaders

        print("Preparing validator")
        if dataset in ["MNIST", "FashionMNIST"]:
            from vae_experiments.evaluation_models.lenet import Model
            net = Model()
            model_path = "vae_experiments/evaluation_models/lenet"  # + dataset
            net.load_state_dict(torch.load(model_path))
            net.to(device)
            net.eval()
            self.dims = 84  # 128
            self.score_model_func = net.part_forward
        elif dataset.lower() == "celeba":
            from vae_experiments.evaluation_models.inception import InceptionV3
            self.dims = 2048
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
            model = InceptionV3([block_idx])
            if score_model_device:
                model = model.to(score_model_device)
            model.eval()
            self.score_model_func = lambda batch: model(batch)[0]
        self.stats_file_name = f"{stats_file_name}_dims_{self.dims}"

    def compute_fid(self, curr_global_decoder, class_table, task_id, translate_noise=True, starting_point=None):
        curr_global_decoder.eval()
        class_table = class_table[:task_id + 1]
        test_loader = self.dataloaders[task_id]
        with torch.no_grad():
            distribution_orig = []
            distribution_gen = []
            task_samplers = []
            # This is the other way round, we first select class from the class table, and then we randomly sample
            # from tasks where this class was present

            for class_id in range(self.n_classes):
                local_probs = class_table[:, class_id] * 1.0 / torch.sum(class_table[:, class_id])
                task_samplers.append(torch.distributions.categorical.Categorical(probs=local_probs))

            precalculated_statistics = False
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
                tasks_sampled = []
                y = y.sort()[0]
                labels, counts = torch.unique_consecutive(y, return_counts=True)
                for i, n_occ in zip(labels, counts):
                    tasks_sampled.append(task_samplers[i].sample([n_occ]))

                task_ids = torch.cat(tasks_sampled)
                if starting_point != None:
                    task_ids = torch.zeros(len(task_ids)) + starting_point
                example = generate_images(curr_global_decoder, z, bin_z, task_ids, y, translate_noise=translate_noise)
                if not precalculated_statistics:
                    distribution_orig.append(self.score_model_func(x).cpu().detach().numpy())
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

    def compute_fid_from_examples(self, args, generations, task_id):
        distribution_orig = []
        precalculated_statistics = False
        stats_file_path = f"results/orig_stats/compare_files_{args.dataset}_{args.experiment_name}_{task_id}.npy"
        test_loader = self.dataloaders[task_id]

        if os.path.exists(stats_file_path):
            print(f"Loading cached original data statistics from: {self.stats_file_name}")
            distribution_orig = np.load(stats_file_path)
            precalculated_statistics = True
        print("Calculating FID:")
        if not precalculated_statistics:
            for idx, batch in enumerate(test_loader):
                x = batch[0].to(self.device)
                distribution_orig.append(self.score_model_func(x).cpu().detach().numpy())

        if args.dataset.lower() == "mnist":
            generations = generations.reshape(-1, 1, 28, 28)
        elif args.dataset.lower() == "celeba":
            generations = generations.reshape(-1, 3, 64, 64)
        generations = torch.from_numpy(generations).to(self.device)
        distribution_gen = self.score_model_func(generations).cpu().detach().numpy().reshape(-1, self.dims)

        if not precalculated_statistics:
            distribution_orig = np.array(np.concatenate(distribution_orig)).reshape(-1, self.dims)
            np.save(stats_file_path, distribution_orig)
        print(f"Orig:{len(distribution_orig)}, Gen:{len(distribution_gen)}")
        precision, recall = compute_prd_from_embedding(
            eval_data=distribution_orig,
            ref_data=distribution_gen[np.random.choice(len(distribution_gen), len(distribution_orig), False)])
        precision, recall = prd_to_max_f_beta_pair(precision, recall)
        print(f"Precision:{precision},recall: {recall}")

        return calculate_frechet_distance(distribution_gen, distribution_orig), precision, recall

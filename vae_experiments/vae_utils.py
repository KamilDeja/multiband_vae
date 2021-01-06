import os

from mpl_toolkits.axes_grid1 import ImageGrid
# from vae_experiments.models_definition import unpackbits
import matplotlib.pyplot as plt
import numpy as np
import torch
from vae_experiments.fid import calculate_frechet_distance


def prepare_class_samplres(task_id, class_table):
    ########### Maybe compute only once and pass to the function?
    class_samplers = []
    for task_id in range(task_id):
        local_probs = class_table[task_id] * 1.0 / torch.sum(class_table[task_id])
        class_samplers.append(torch.distributions.categorical.Categorical(probs=local_probs))
    return class_samplers


def plot_results(experiment_name, curr_global_decoder, class_table, n_tasks, n_img=5, suffix=""):
    curr_global_decoder.eval()
    z = torch.randn([n_img * (n_tasks + 1), curr_global_decoder.latent_size]).to(curr_global_decoder.device)
    task_ids = np.repeat(list(range(n_tasks + 1)), n_img)
    task_ids = torch.from_numpy(task_ids).float()
    class_samplers = prepare_class_samplres(n_tasks + 1, class_table)

    sampled_classes = []
    for i in range(n_tasks + 1):  ## Including current class
        sampled_classes.append(class_samplers[i].sample([n_img]))
    sampled_classes = torch.cat(sampled_classes)
    example = generate_images(curr_global_decoder, z, task_ids, sampled_classes)
    example = example.cpu().detach().numpy()
    fig = plt.figure(figsize=(10., 10.))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(n_tasks + 1, n_img),
                     axes_pad=0.5,
                     )

    for ax, im, target in zip(grid, example, sampled_classes):
        im = np.swapaxes(im, 0, 2)
        im = np.swapaxes(im, 0, 1)
        ax.imshow(im.squeeze())
        ax.set_title('Class: {}'.format(target))

    plt.savefig("results/" + experiment_name + "/generations_task_" + str(n_tasks) + suffix)
    plt.close()


def generate_images(curr_global_decoder, z, task_ids, y):
    example = curr_global_decoder(z, task_ids, y)
    return example


def generate_noise_for_previous_data(n_img, n_task, latent_size, same_z=False):
    if same_z:
        z = torch.randn([n_img // (n_task + 1), latent_size]).repeat([n_task + 1])
        raise NotImplementedError  # Check first if it works
    else:
        z = torch.randn([n_img, latent_size])
    return z


def generate_previous_data(curr_global_decoder, class_table, n_tasks, n_img, same_z=False, return_z = False):
    with torch.no_grad():
        curr_class_table = class_table[:n_tasks]
        z = generate_noise_for_previous_data(n_img, n_tasks, curr_global_decoder.latent_size, same_z).to(
            curr_global_decoder.device)
        tasks_dist = torch.sum(curr_class_table, dim=1) * n_img // torch.sum(curr_class_table)
        tasks_dist[0:n_img - tasks_dist.sum()] += 1  # To fix the division
        assert sum(tasks_dist) == n_img
        task_ids = []
        for task_id in range(n_tasks):
            if tasks_dist[task_id] > 0:
                task_ids.append([task_id] * tasks_dist[task_id])
        task_ids = torch.from_numpy(np.concatenate(task_ids)).float()
        assert len(task_ids) == n_img

        class_samplers = prepare_class_samplres(n_tasks, curr_class_table)

        sampled_classes = []
        for task_id in range(n_tasks):
            if tasks_dist[task_id]>0:
                sampled_classes.append(class_samplers[task_id].sample(tasks_dist[task_id].view(-1, 1)))
        sampled_classes = torch.cat(sampled_classes)
        assert len(sampled_classes) == n_img

        example = generate_images(curr_global_decoder, z, task_ids, sampled_classes)
        if return_z:
            return example, sampled_classes, z, task_ids
        else:
            return example, sampled_classes

from mpl_toolkits.axes_grid1 import ImageGrid
from vae_experiments.models_definition import unpackbits
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_results(experiment_name, curr_global_decoder, n_tasks, total_n_codes, global_n_codes,
                 global_classes_list, n_img=5, suffix=""):
    curr_global_decoder.eval()
    current_starts = [100]*(n_tasks+1)
    example, classes, _ = generate_previous_data(curr_global_decoder, n_tasks + 1, n_img, current_starts,
                                                 global_classes_list, total_n_codes, global_n_codes)
    example = example.cpu().detach().numpy()
    fig = plt.figure(figsize=(10., 10.))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(n_tasks + 1, n_img),
                     axes_pad=0.5,
                     )

    for ax, im, target in zip(grid, example, classes):
        im = np.swapaxes(im, 0, 2)
        im = np.swapaxes(im, 0, 1)
        ax.imshow(im.squeeze())
        ax.set_title('Class: {}'.format(target))

    plt.savefig("results/" + experiment_name + "/generations_task_" + str(n_tasks) + suffix)
    plt.close()


def generate_codes_task(task_id, n_codes, global_n_codes, current_start, curr_global_decoder,
                        global_classes_list):
    exp_values = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
    parts = len(exp_values)
    start_id = int(np.sum(global_n_codes[:task_id]))
    codes_rep = torch.Tensor()
    codes_range = range(start_id + current_start,
                        min(start_id + current_start + n_codes, start_id + global_n_codes[task_id]))
    for exp_value in exp_values:
        codes = codes_range * np.array(
            exp_value ** np.floor(curr_global_decoder.latent_size // parts * np.log(2) / np.log(exp_value)),
            dtype=np.longlong) % 2 ** curr_global_decoder.latent_size // parts
        codes = torch.tensor(
            unpackbits(np.array(codes, dtype=np.longlong), curr_global_decoder.latent_size // parts)).float()
        codes_rep = torch.cat([codes_rep, codes], 1)

    selected_classes = (global_classes_list[
                        int(start_id + current_start):int(
                            min(start_id + current_start + n_codes, start_id + global_n_codes[task_id]))])
    return codes_rep, selected_classes


def generate_codes(n_tasks, n_codes, total_n_codes, global_n_codes, current_starts, curr_global_decoder,
                   global_classes_list):
    codes_list = []
    selected_classes = []
    task_ids = []

    for task_id in range(n_tasks):
        codes_rep, classes = generate_codes_task(task_id, n_codes, global_n_codes, current_starts[task_id], curr_global_decoder,
                                                 global_classes_list)
        codes_list.append(codes_rep)
        selected_classes.append(classes)
        task_ids.append([task_id] * len(classes))

    codes = torch.cat(codes_list)
    classes = torch.from_numpy(np.concatenate(selected_classes)).long()
    return codes, classes, np.concatenate(task_ids)


def generate_previous_data(curr_global_decoder, n_tasks, n_img, current_start, global_classes_list,
                           total_n_codes, global_n_codes, return_codes=False):
    if not n_tasks:
        return torch.Tensor(), torch.Tensor(), torch.Tensor()

    curr_global_decoder.eval()
    # global_classes_list = np.array(global_classes_list)
    codes, classes, task_ids = generate_codes(n_tasks, n_img, total_n_codes, global_n_codes, current_start,
                                              curr_global_decoder,
                                              global_classes_list)
    codes_rep = (codes.repeat([1, 1]).to(curr_global_decoder.device) * 2 - 1)
    # task_ids = np.repeat(list(range(n_tasks)), n_img)
    if len(codes_rep) == 0:
        return torch.Tensor(), torch.Tensor(), torch.Tensor()
    example = torch.sigmoid(curr_global_decoder(codes_rep, task_ids))
    if return_codes:
        return example, classes, task_ids, codes_rep
    return example, classes, task_ids


def generate_current_data(curr_global_decoder, task_id, n_img, current_start, global_classes_list,
                          total_n_codes, global_n_codes):

    curr_global_decoder.eval()

    codes, selected_classes = generate_codes_task(task_id, n_img, global_n_codes, current_start, curr_global_decoder,
                                                  global_classes_list)

    classes = torch.from_numpy(np.array(selected_classes)).long()

    codes_rep = (codes.repeat([1, 1]).to(curr_global_decoder.device) * 2 - 1)
    task_ids = np.array([task_id]*len(selected_classes))
    example = torch.sigmoid(curr_global_decoder(codes_rep, task_ids))
    return example, classes, task_ids


def generate_previous_and_current_data(curr_global_decoder, n_tasks, n_img, current_start, global_classes_list,
                                       total_n_codes, global_n_codes):
    return generate_previous_data(curr_global_decoder, n_tasks + 1, n_img, current_start, global_classes_list,
                                  total_n_codes, global_n_codes)

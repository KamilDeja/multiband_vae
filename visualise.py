import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import os


def dict2array(results):
    runs = len(results)
    tasks = len(results[0])
    array = np.zeros((runs, tasks, tasks))
    for run, dict_run in results.items():
        for e, (key, val) in enumerate(reversed(dict_run.items())):
            for e1, (k, v) in enumerate(reversed(val.items())):
                array[int(run), tasks - int(e1) - 1, tasks - int(e) - 1] = round(v, 3)
    return np.transpose((array), axes=(0, 2, 1))


def grid_plot(ax, array, exp_name, type):
    if type == "fid":
        round = 1
    else:
        round = 2
    avg_array = np.around(np.mean(array, axis=0), round)
    num_tasks = array.shape[1]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#287233", "#4c1c24"])
    ax.imshow(avg_array, vmin=50, vmax=300, cmap=cmap)
    for i in range(len(avg_array)):
        for j in range(avg_array.shape[1]):
            if j >= i:
                ax.text(j, i, avg_array[i, j], va='center', ha='center', c='w', fontsize=70 / num_tasks)
    ax.set_yticks(np.arange(num_tasks))
    ax.set_ylabel('Number of tasks')
    ax.set_xticks(np.arange(num_tasks))
    ax.set_xlabel('Tasks finished')
    ax.set_title(
        f"{type} -- {np.round(np.mean(array[:, :, -1], axis=(0, 1)), 3)} -- std {np.round(np.std(np.mean(array[:, :, -1], axis=1), axis=0), 2)}")


def acc_over_time_plot(ax, array):
    num_tasks = array.shape[1]
    acc_over_time = np.sum(array, axis=1) / np.arange(1, num_tasks + 1)
    mean, std = np.mean(acc_over_time, axis=0), np.std(acc_over_time, axis=0)
    ax.fill_between(np.arange(1, num_tasks + 1), mean - std, mean + std, alpha=0.3)
    ax.plot(np.arange(1, num_tasks + 1), mean)


def plot_final_results(names, rpath='results/', type="fid", fid_local_vae=None):
    fig = plt.figure(figsize=(13, 5 * len(names)))
    gs = GridSpec(len(names), 3)
    additional = ""
    if fid_local_vae != None:
        additional = "local_vae_fid: " + str([(x, round(fid_local_vae[x], 2)) for x in fid_local_vae])
    fig.suptitle(f"Experiment: {names[0]}\n {additional}")
    for e, name in enumerate(names):
        acc_dict = np.load(f"{rpath}{name}/fid.npy", allow_pickle=True).item()
        arr_fid = dict2array(acc_dict)
        acc_dict = np.load(f"{rpath}{name}/precision.npy", allow_pickle=True).item()
        arr_prec = dict2array(acc_dict)
        acc_dict = np.load(f"{rpath}{name}/recall.npy", allow_pickle=True).item()
        arr_rec = dict2array(acc_dict)
        ax1 = fig.add_subplot(gs[e, 0])
        ax2 = fig.add_subplot(gs[e, 1])
        ax3 = fig.add_subplot(gs[e, 2])
        grid_plot(ax1, arr_fid, name, "fid")
        grid_plot(ax2, arr_prec, name, "precision")
        grid_plot(ax3, arr_rec, name, "recall")
        # acc_over_time_plot(ax2, arr)

    # plt.show()
    plt.savefig(rpath + names[0] + f"/results_visualisation")#, dpi=200)


if __name__ == '__main__':
    plot_final_results(['CelebA_50_16_fixed_eval_dirichlet_1_sim09_warmup5_bin8'])

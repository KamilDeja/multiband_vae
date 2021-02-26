import os

import umap
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from vae_experiments.vae_utils import *


class Visualizer:
    def __init__(self, decoder, class_table, task_id, experiment_name, n_init_samples=1000, same_nr_per_task=True):
        self.task_id = task_id + 1
        if same_nr_per_task:
            class_table[:task_id + 1] = 1
        self.class_table = class_table
        recon_prev, classes_prev, z_prev, task_ids_prev, embeddings_prev = generate_previous_data(decoder,
                                                                                                  class_table=class_table,
                                                                                                  n_tasks=self.task_id,
                                                                                                  n_img=n_init_samples,
                                                                                                  return_z=True,
                                                                                                  num_local=n_init_samples // task_id)
        self.umap = umap.UMAP()
        self.umap.fit(embeddings_prev.cpu())
        self.selected_images = list(range(0, 1000, 50))
        save_dir = f"results/{experiment_name}/latent_images/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def visualize_latent(self, decoder, epoch_n, experiment_name, n_samples=1000):
        recon_prev, classes_prev, z_prev, task_ids_prev, embeddings_prev = generate_previous_data(decoder,
                                                                                                  class_table=self.class_table,
                                                                                                  n_tasks=self.task_id,
                                                                                                  n_img=n_samples,
                                                                                                  return_z=True,
                                                                                                  num_local=n_samples // self.task_id)

        x_embedded = self.umap.transform(embeddings_prev.cpu())
        noises_to_plot = pd.DataFrame(x_embedded)
        noises_to_plot["batch"] = task_ids_prev.cpu().detach().numpy()

        examples = []
        examples_locations = []
        for i in self.selected_images:
            noise_tmp = z_prev[0][i].view(1, -1)
            bin_tmp = z_prev[1][i].view(1, -1)
            task_id_tmp = task_ids_prev[i].view(1, -1)
            examples.append(decoder(noise_tmp, bin_tmp, task_id_tmp, None).detach().cpu().numpy().squeeze())
            examples_locations.append(noises_to_plot.iloc[i])

        fig, ax = plt.subplots(figsize=(15, 10))
        # ax.scatter(noises_to_plot_tsne[0],noises_to_plot_tsne[1],c=noises_to_plot_tsne["batch"],s=3,alpha=0.8)
        sns.scatterplot(
            x=0, y=1,
            hue="batch",
            palette=sns.color_palette("hls", self.task_id),
            data=noises_to_plot,
            legend="full",
            alpha=0.9
        )
        # plt.imshow(example)
        for location, example in zip(examples_locations, examples):
            x, y = location[0], location[1]
            batch = int(location["batch"])
            ab = AnnotationBbox(OffsetImage(example), (x, y), frameon=True,
                                bboxprops=dict(edgecolor=sns.color_palette("hls", self.task_id)[batch], width=10))
            ax.add_artist(ab)

        plt.title(f"Latent visualisation epoch {epoch_n}", fontsize=34)
        plt.savefig(f"results/{experiment_name}/latent_images/task_{self.task_id-1}_epoch_{epoch_n}")
        plt.close()
        # plt.show()

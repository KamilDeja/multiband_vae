import os

import umap
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from vae_experiments.vae_utils import *


class Visualizer:
    def __init__(self, decoder, class_table, task_id, experiment_name, n_init_samples=1000, same_nr_per_task=True):
        self.task_id = task_id
        if same_nr_per_task:
            class_table[:task_id + 1] = 1
        self.class_table = class_table
        # recon_prev, classes_prev, z_prev, task_ids_prev, embeddings_prev = generate_previous_data(decoder,
        #                                                                                           class_table=class_table,
        #                                                                                           n_tasks=self.task_id + 1,
        #                                                                                           n_img=n_init_samples,
        #                                                                                           return_z=True,
        #                                                                                           num_local=n_init_samples // task_id)
        self.umap = umap.UMAP(metric="cosine", n_neighbors=100)
        # embeddings = torch.cat(embeddings_prev, embeddings_curr)
        # self.umap.fit(embeddings_prev.cpu())
        self.selected_images = list(range(0, 1000, 200))
        save_dir = f"results/{experiment_name}/latent_images/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def visualize_latent(self, encoder, decoder, epoch_n, experiment_name, orig_images, orig_labels, n_samples=1000):
        recon_prev, classes_prev, z_prev, task_ids_prev, embeddings_prev = generate_previous_data(decoder,
                                                                                                  class_table=self.class_table,
                                                                                                  n_tasks=self.task_id,
                                                                                                  n_img=n_samples,
                                                                                                  return_z=True,
                                                                                                  num_local=n_samples // self.task_id)

        orig_images = orig_images[:n_samples], orig_labels[:n_samples]
        means, log_var, bin_z = encoder(orig_images[0].to(encoder.device),
                                        orig_images[1].to(encoder.device))
        std = torch.exp(0.5 * log_var)
        binary_out = torch.distributions.Bernoulli(logits=bin_z).sample()
        z_bin_current_compare = binary_out * 2 - 1
        eps = torch.randn([len(orig_images[0]), decoder.latent_size]).to(encoder.device)
        z_current_compare = eps * std + means
        task_ids_current_compare = torch.zeros(len(orig_images[0])) + self.task_id
        task_ids = torch.cat([task_ids_prev, task_ids_current_compare])
        embeddings_curr = decoder.translator(z_current_compare, z_bin_current_compare, task_ids_current_compare)
        embeddings = torch.cat([embeddings_prev, embeddings_curr]).cpu().detach()
        x_embedded = self.umap.fit_transform(embeddings.cpu())
        noises_to_plot = pd.DataFrame(x_embedded)
        noises_to_plot["batch"] = task_ids.cpu().detach().numpy()

        examples = []
        examples_locations = []
        for i in self.selected_images:
            noise_tmp = z_prev[0][i].view(1, -1)
            bin_tmp = z_prev[1][i].view(1, -1)
            task_id_tmp = task_ids_prev[i].view(1, -1)
            examples.append(decoder(noise_tmp, bin_tmp, task_id_tmp, None).detach().cpu().numpy().squeeze())
            examples_locations.append(noises_to_plot.iloc[i])

        for i in range(len(self.selected_images) // self.task_id):
            examples.append(orig_images[0][i].detach().cpu().numpy().squeeze())
            # print(len(task_ids_prev),len(noises_to_plot))
            examples_locations.append(noises_to_plot.iloc[len(task_ids_prev) + i])

        fig, ax = plt.subplots(figsize=(15, 10))
        # ax.scatter(noises_to_plot_tsne[0],noises_to_plot_tsne[1],c=noises_to_plot_tsne["batch"],s=3,alpha=0.8)
        sns.scatterplot(
            x=0, y=1,
            hue="batch",
            palette=sns.color_palette("hls", 3)[:self.task_id + 1],
            data=noises_to_plot,
            legend="full",
            alpha=0.9
        )
        # plt.imshow(example)
        for location, example in zip(examples_locations, examples):
            x, y = location[0], location[1]
            batch = int(location["batch"])
            ab = AnnotationBbox(OffsetImage(example, cmap='Greys', zoom=2), (x, y), frameon=True,
                                bboxprops=dict(facecolor=sns.color_palette("hls", 3)[batch], width=10))
            ax.add_artist(ab)

        plt.title(f"Latent visualisation epoch {epoch_n}", fontsize=34)
        plt.savefig(f"results/{experiment_name}/latent_images/task_{self.task_id}_epoch_{epoch_n}")
        plt.close()
        # plt.show()

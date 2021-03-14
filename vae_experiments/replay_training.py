import copy
import time
import numpy as np
import torch
import torch.functional as F
from torch import optim, nn

from vae_experiments.latent_visualise import Visualizer
from vae_experiments.vae_utils import generate_previous_data
from vae_experiments.training_functions import loss_fn, cosine_distance


def train_with_replay(args, local_vae, task_loader, train_dataset_loader_big, task_id, class_table, train_same_z=True):
    limit_previous_examples, warmup_rounds, noise_diff_threshold = args.limit_previous, args.global_warmup, args.cosine_sim
    optimizer = torch.optim.Adam(list(local_vae.parameters()), lr=args.local_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    frozen_model = copy.deepcopy(local_vae.decoder)
    frozen_model.eval()
    local_vae.train()
    table_tmp = torch.zeros(class_table.size(1), dtype=torch.long)
    task_ids = task_id
    ones_distribution = torch.zeros([local_vae.binary_latent_size]).to(local_vae.device)
    total_examples = 0
    if (task_id > 0) and args.visualise_latent:
        local_vae.decoder.ones_distribution = torch.cat([local_vae.decoder.ones_distribution,
                                                         (ones_distribution.cpu().detach()).view(1, -1) + 0.5], 0)
        visualizer = Visualizer(local_vae.decoder, class_table, task_id=task_id, experiment_name=args.experiment_name)
    if local_vae.in_size == 28:
        marginal_loss = nn.BCELoss(reduction="sum")
    else:
        marginal_loss = nn.MSELoss(reduction="sum")
    for epoch in range(args.gen_ae_epochs):
        if task_id > 0:
            orig_images = next(iter(train_dataset_loader_big))
            orig_images, orig_labels = orig_images[0].to(local_vae.device), orig_images[1]

        if (task_id > 0) and args.visualise_latent:
            visualizer.visualize_latent(local_vae.encoder, local_vae.decoder, epoch_n=epoch, orig_images=orig_images,
                                        orig_labels=orig_labels, experiment_name=args.experiment_name)
            ones_distribution = torch.zeros([local_vae.binary_latent_size]).to(local_vae.device)
        losses = []
        sum_changed = torch.zeros([task_id + 1])
        start = time.time()
        gumbel_temp = max(1 - (5 * epoch / args.gen_ae_epochs), 0.01)
        if gumbel_temp < 0.1:
            gumbel_temp = None

        for iteration, batch in enumerate(task_loader):
            x = batch[0].to(local_vae.device)
            y = batch[1].to(local_vae.device)

            if epoch == 0:
                class_counter = torch.unique(y, return_counts=True)
                table_tmp[class_counter[0]] += class_counter[1].cpu()

            if task_id > 0:
                n_prev_examples = int(len(x) * min(task_id, 3) * limit_previous_examples)
                recon_prev, recon_classes, z_prev, task_ids_prev, encodings = generate_previous_data(frozen_model,
                                                                                                     class_table=class_table,
                                                                                                     n_tasks=task_id,
                                                                                                     n_img=n_prev_examples,
                                                                                                     translate_noise=True,
                                                                                                     return_z=True,
                                                                                                     same_z=train_same_z)


                task_ids = torch.cat([torch.zeros(x.size(0)) + task_id, task_ids_prev], dim=0)
                x = torch.cat([x, recon_prev], dim=0)
                y = torch.cat([y.view(-1), recon_classes.to(local_vae.device)], dim=0)

            recon_x, mean, log_var, z, binary_out, bin_prob = local_vae(x, task_ids, y, temp=gumbel_temp,
                                                                        translate_noise=True)

            loss, kl_div = loss_fn(recon_x, x, mean, log_var, marginal_loss)  # + bin_loss_fn(binary_out)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(local_vae.parameters(), 5.0)
            optimizer.step()

            losses.append(loss.item())

            if args.visualise_latent or (epoch == args.gen_ae_epochs - 1):
                ones_distribution += (binary_out / 2 + 0.5).sum(0)
                total_examples += len(binary_out)
        scheduler.step()
        #     print("lr:",scheduler.get_lr())
        if task_id > 0 and args.visualise_latent:
            local_vae.decoder.ones_distribution[task_id] = (ones_distribution.cpu().detach() / total_examples).view(1,
                                                                                                                    -1)
            ones_distribution = torch.zeros([local_vae.binary_latent_size]).to(local_vae.device)
        if epoch % 1 == 0:
            print("Epoch: {}/{}, loss: {}, last epoch took {} s".format(epoch, args.gen_ae_epochs, np.mean(losses),
                                                                        time.time() - start))
            if sum_changed.sum() > 0:
                print(
                    f"Epoch: {epoch} - changing from batches: {[(idx, n_changes) for idx, n_changes in enumerate(sum_changed.tolist())]}")
    if local_vae.decoder.ones_distribution == None:
        local_vae.decoder.ones_distribution = (ones_distribution.cpu().detach() / total_examples).view(1, -1)
    else:
        if not args.visualise_latent:
            local_vae.decoder.ones_distribution = torch.cat([local_vae.decoder.ones_distribution,
                                                             (ones_distribution.cpu().detach() / total_examples).view(1,
                                                                                                                      -1)],
                                                            0)
    return local_vae.decoder, table_tmp

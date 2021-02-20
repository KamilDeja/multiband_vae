import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from geomloss import SamplesLoss

from vae_experiments.lap_loss import LapLoss
from vae_experiments.vae_utils import *
import copy

mse_loss = nn.MSELoss(reduction="sum")
bce_loss = nn.BCELoss(reduction="sum")
sinkhorn_loss = SamplesLoss("sinkhorn", blur=0.05, scaling=0.95, diameter=0.01, debias=True)


def loss_fn(y, x_target, mu, sigma, lap_loss_fn=None):
    marginal_likelihood = bce_loss(y, x_target) / y.size(0)
    # marginal_likelihood = F.binary_cross_entropy(y, x_target, reduction="none")
    # marginal_likelihood = torch.sum(marginal_likelihood, dim=[2, 3])
    # marginal_likelihood = torch.mean(marginal_likelihood, dim=[2, 3])  # / y.size(0)
    # marginal_likelihood = torch.mean(marginal_likelihood)
    # print(marginal_likelihood)
    # F.binary_cross_entropy(y, x_target, reduction='sum') / y.size(0)
    # marginal_likelihood = mse_loss(y, x_target) / y.size(0)
    KL_divergence = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp()) / y.size(0)
    # KL_divergence = torch.mean(KL_divergence) / y.size(1)  # / y.size(0)  # / y.size(1)
    if lap_loss_fn:
        lap_loss = lap_loss_fn(y, x_target)
        loss = marginal_likelihood + x_target[0].size()[1] * x_target[0].size()[1] * lap_loss + KL_divergence
    else:
        loss = marginal_likelihood + KL_divergence

    return loss


def bin_loss_fn(x):
    utilization_loss = x.mean(0).pow(2).sum()  # (x.mean()).pow(2)  # Mean should be zero
    # x = x / 2 + 0.5
    # with torch.no_grad():
    #     targets = torch.round(x)
    # # print(x)
    # binarizatison_loss = bce_loss(x, targets) / x.size(0)
    # print(f"Utilization_loss {utilization_loss}")# Binarization_loss {binarization_loss}")
    return 0  # 50 * utilization_loss  # binarization_loss#utilization_loss + binarization_loss


def find_best_starting_point(local_vae, task_loader, task_id):
    if task_id == 0:
        return 1, 0
    if task_id == 1:
        return 1, 0
    with torch.no_grad():
        loss = nn.MSELoss()
        losses = torch.zeros(task_id)
        for iteration, batch in enumerate(task_loader):
            x = batch[0].to(local_vae.device)
            y = batch[1]  # .to(local_vae.device)
            batch_size = len(x)
            noise = torch.randn([batch_size, local_vae.latent_size]).to(local_vae.device).repeat([task_id, 1])
            task_ids = torch.Tensor([[task] * batch_size for task in range(task_id)]).view(-1)
            recon_x, _, _, _, _ = local_vae(x.repeat([task_id, 1, 1, 1]), task_ids, y.repeat([task_id]), temp=1,
                                            translate_noise=True, noise=noise)
            for task in range(task_id):
                losses[task] += (loss(x, recon_x[task * batch_size:(task + 1) * batch_size]))
            if iteration > 10:
                break
        print(f"Starting points_losses:{losses}")
    return torch.min(losses), torch.argmin(losses).item()


def train_local_generator(local_vae, task_loader, task_id, n_classes, n_epochs=100, use_lap_loss=False,
                          local_start_lr=0.001):
    local_vae.train()
    local_vae.translator.train()
    # if task_id == 0:
    #     translate_noise = False
    # else:
    translate_noise = True
    min_loss, starting_point = find_best_starting_point(local_vae, task_loader, task_id)
    print(f"Selected {starting_point} as staring point for task {task_id}")
    local_vae.starting_point = starting_point
    lr = min_loss * local_start_lr
    print(f"lr set to: {lr}")
    scheduler_rate = 1  # 0.99
    optimizer = torch.optim.Adam(list(local_vae.parameters()) + list(local_vae.translator.parameters()), lr=lr,
                                 betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)
    table_tmp = torch.zeros(n_classes, dtype=torch.long)
    lap_loss = LapLoss(device=local_vae.device) if use_lap_loss else None
    if task_id > 0:
        optimizer = torch.optim.Adam(local_vae.encoder.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)

    for epoch in range(n_epochs):
        losses = []
        start = time.time()
        if (task_id != 0) and (epoch == min(10, max(n_epochs // 10, 5))):
            optimizer = torch.optim.Adam(list(local_vae.parameters()) + list(local_vae.translator.parameters()),
                                         lr=lr)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)
            # for name, p in local_vae.named_parameters():
            #     p.requires_grad = True
        gumbel_temp = max(1 - (5 * epoch / (n_epochs)), 0.01)
        if gumbel_temp < 0.1:
            gumbel_temp = None
        if epoch == n_epochs - 1:
            ones_distribution = torch.zeros([local_vae.binary_latent_size]).to(local_vae.device)
            total_examples = 0
        for iteration, batch in enumerate(task_loader):

            x = batch[0].to(local_vae.device)
            y = batch[1]  # .to(local_vae.device)
            recon_x, mean, log_var, z, bin_x = local_vae(x, starting_point, y, temp=gumbel_temp,
                                                         translate_noise=translate_noise)

            loss = loss_fn(recon_x, x, mean, log_var, lap_loss)
            binary_loss = bin_loss_fn(bin_x)
            loss_final = loss + binary_loss
            optimizer.zero_grad()
            loss_final.backward()
            optimizer.step()

            losses.append(loss.item())
            if epoch == 0:
                class_counter = torch.unique(y, return_counts=True)
                table_tmp[class_counter[0]] += class_counter[1].cpu()
            if epoch == n_epochs - 1:
                ones_distribution += (bin_x / 2 + 0.5).sum(0)
                total_examples += len(bin_x)
        scheduler.step()
        #     print("lr:",scheduler.get_lr())
        #     print(iteration,len(task_loader))
        if epoch % 1 == 0:
            print("Epoch: {}/{}, loss: {}, took: {} s".format(epoch, n_epochs, np.mean(losses), time.time() - start))
    local_vae.decoder.ones_distribution[task_id] = ones_distribution.cpu().detach() / total_examples
    return table_tmp


def train_global_decoder(curr_global_decoder, local_vae, task_id, class_table,
                         models_definition, cosine_sim, n_epochs=100, n_iterations=30, batch_size=1000,
                         train_same_z=True,
                         new_global_decoder=False, global_lr=0.0001, limit_previous_examples=0.5):
    if new_global_decoder:
        global_decoder = models_definition.Decoder(latent_size=curr_global_decoder.latent_size, d=curr_global_decoder.d,
                                                   p_coding=curr_global_decoder.p_coding,
                                                   n_dim_coding=curr_global_decoder.n_dim_coding,
                                                   cond_p_coding=curr_global_decoder.cond_p_coding,
                                                   cond_n_dim_coding=curr_global_decoder.cond_n_dim_coding,
                                                   cond_dim=curr_global_decoder.cond_dim,
                                                   device=curr_global_decoder.device,
                                                   translator=models_definition.Translator(
                                                       curr_global_decoder.n_dim_coding, curr_global_decoder.p_coding,
                                                       curr_global_decoder.latent_size, curr_global_decoder.device),
                                                   standard_embeddings=curr_global_decoder.standard_embeddings,
                                                   trainable_embeddings=curr_global_decoder.trainable_embeddings,
                                                   in_size=curr_global_decoder.in_size
                                                   ).to(curr_global_decoder.device)
    else:
        global_decoder = copy.deepcopy(curr_global_decoder)
    curr_global_decoder.eval()
    curr_global_decoder.translator.eval()
    local_vae.eval()
    global_decoder.train()
    global_decoder.ones_distribution = local_vae.decoder.ones_distribution
    curr_global_decoder.ones_distribution = local_vae.decoder.ones_distribution
    # local_vae.translator.train()
    # frozen_translator = copy.deepcopy(curr_global_decoder.translator)
    # frozen_translator.eval()
    optimizer = torch.optim.Adam(list(global_decoder.parameters()) + list(global_decoder.translator.parameters()),
                                 lr=global_lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    criterion = nn.MSELoss(reduction='sum')
    # criterion = nn.BCELoss(reduction='sum')
    embedding_loss_criterion = nn.MSELoss(reduction='sum')
    class_samplers = prepare_class_samplres(task_id + 1, class_table)
    local_starting_point = torch.zeros([batch_size]) + local_vae.starting_point
    task_ids_local = torch.zeros([batch_size]) + task_id
    for epoch in range(n_epochs):
        losses = []
        start = time.time()
        sum_changed = torch.zeros([task_id + 1])
        for iteration in range(n_iterations):
            # Building dataset from previous global model and local model
            recon_prev, classes_prev, z_prev, task_ids_prev, embeddings_prev = generate_previous_data(
                curr_global_decoder,
                class_table=class_table,
                n_tasks=task_id,
                n_img=int(batch_size * task_id * limit_previous_examples),
                return_z=True,
                translate_noise=True,
                same_z=train_same_z)

            if train_same_z:
                z_prev, z_max, z_bin_prev, z_bin_max = z_prev
                z_local = z_max[:batch_size]
                z_bin_local = z_bin_max[:batch_size]

            with torch.no_grad():
                sampled_classes_local = class_samplers[-1].sample([batch_size])
                if not train_same_z:
                    z_local = torch.randn([batch_size, local_vae.latent_size]).to(curr_global_decoder.device)
                    z_bin_local = torch.rand_like(z_local).to(curr_global_decoder.device)
                    z_bin_local = torch.round(z_bin_local) * 2 - 1
                recon_local = local_vae.decoder(z_local, z_bin_local, local_starting_point, sampled_classes_local,
                                                translate_noise=True)

            z_concat = torch.cat([z_prev, z_local])
            z_bin_concat = torch.cat([z_bin_prev, z_bin_local])
            task_ids_concat = torch.cat([task_ids_prev, task_ids_local])
            class_concat = torch.cat([classes_prev, sampled_classes_local])
            cos_sim = torch.nn.CosineSimilarity()
            if epoch > 20:# Warm-up epochs within which we don't switch targets
                with torch.no_grad():
                    prev_noise_translated = global_decoder.translator(z_prev, z_bin_prev, task_ids_prev)
                    # current_noise_translated = local_vae.decoder.translator(z_local, local_starting_point)#@TODO check this?
                    current_noise_translated = global_decoder.translator(z_local, z_bin_local, task_ids_local)
                    noise_diff_threshold = cosine_sim  # current_noise_translated.std() * cosine_sim  # .5  # * 2.5  # 3
                    for prev_task_id in range(task_id):
                        selected_task_ids = torch.where(task_ids_prev == prev_task_id)[0]
                        selected_task_ids = selected_task_ids[:len(current_noise_translated)]
                        noises_distances = cos_sim(prev_noise_translated[selected_task_ids],
                                                   current_noise_translated[:len(selected_task_ids)])
                        if (noises_distances > noise_diff_threshold).sum() > 0:
                            imgs_task = recon_prev[selected_task_ids]
                            z_bin_task = z_bin_prev[selected_task_ids]
                            same_bin_z = (z_bin_task - z_bin_local[:len(z_bin_task)]).abs().sum(1) == 0
                            to_switch = (noises_distances > noise_diff_threshold) * same_bin_z
                            imgs_task[to_switch] = recon_local[:len(selected_task_ids)][to_switch]
                            recon_prev[selected_task_ids] = imgs_task
                            sum_changed[prev_task_id] += to_switch.sum().item()

            recon_concat = torch.cat([recon_prev, recon_local])

            n_mini_batches = math.ceil(len(z_concat) / batch_size)
            shuffle = torch.randperm(len(task_ids_concat))
            z_concat = z_concat[shuffle]
            z_bin_concat = z_bin_concat[shuffle]
            task_ids_concat = task_ids_concat[shuffle]
            recon_concat = recon_concat[shuffle]
            class_concat = class_concat[shuffle]

            for batch_id in range(n_mini_batches):
                start_point = batch_id * batch_size
                end_point = min(len(task_ids_concat), (batch_id + 1) * batch_size)
                global_recon = global_decoder(z_concat[start_point:end_point], z_bin_concat[start_point:end_point],
                                              task_ids_concat[start_point:end_point],
                                              class_concat[start_point:end_point])
                loss = criterion(global_recon, recon_concat[start_point:end_point])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
        scheduler.step()
        if (epoch % 1 == 0):
            print("Epoch: {}/{}, loss: {}, took: {} s".format(epoch, n_epochs, np.mean(losses), time.time() - start))
            if sum_changed.sum() > 0:
                print(
                    f"Epoch: {epoch} - changing from batches: {[(idx, n_changes) for idx, n_changes in enumerate(sum_changed.tolist())]}")
    return global_decoder

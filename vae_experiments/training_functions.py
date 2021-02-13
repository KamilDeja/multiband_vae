import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from vae_experiments.lap_loss import LapLoss
from vae_experiments.vae_utils import *
import copy

mse_loss = nn.MSELoss(reduction="sum")


def loss_fn(y, x_target, mu, sigma, lap_loss_fn=None):
    # marginal_likelihood = F.binary_cross_entropy(y, x_target, reduction='sum') / y.size(0)
    marginal_likelihood = mse_loss(y, x_target) / y.size(0)

    KL_divergence = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp()) / y.size(0)
    if lap_loss_fn:
        lap_loss = lap_loss_fn(y, x_target)
        loss = marginal_likelihood + x_target[0].size()[1] * x_target[0].size()[1] * lap_loss + KL_divergence
    else:
        loss = marginal_likelihood + KL_divergence

    return loss


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
            recon_x, _, _, _ = local_vae(x.repeat([task_id, 1, 1, 1]), task_ids, y.repeat([task_id]),
                                         translate_noise=True, noise=noise)
            for task in range(task_id):
                losses[task] += (loss(x, recon_x[task * batch_size:(task + 1) * batch_size]))
            if iteration > 10:
                break
        print(f"Starting points_losses:{losses}")
    return torch.min(losses), torch.argmin(losses).item()


def train_local_generator(local_vae, task_loader, task_id, n_classes, n_epochs=100, use_lap_loss=False):
    local_vae.train()
    local_vae.translator.train()
    # if task_id == 0:
    #     translate_noise = False
    # else:
    translate_noise = True
    min_loss, starting_point = find_best_starting_point(local_vae, task_loader, task_id)
    print(f"Selected {starting_point} as staring point for task {task_id}")
    local_vae.starting_point = starting_point
    lr = min_loss * 0.001
    print(f"lr set to: {lr}")
    scheduler_rate = 0.95
    optimizer = torch.optim.Adam(list(local_vae.parameters()) + list(local_vae.translator.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)
    table_tmp = torch.zeros(n_classes, dtype=torch.long)
    lap_loss = LapLoss(device=local_vae.device) if use_lap_loss else None
    if task_id > 0:
        optimizer = torch.optim.Adam(local_vae.encoder.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)
    #     local_vae.decoder.eval()
    #     for name, p in local_vae.named_parameters():
    #         p.requires_grad = False

    for epoch in range(n_epochs):
        losses = []
        start = time.time()
        if epoch == 10:
            optimizer = torch.optim.Adam(list(local_vae.parameters()) + list(local_vae.translator.parameters()),
                                         lr=lr)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)
            # for name, p in local_vae.named_parameters():
            #     p.requires_grad = True
        for iteration, batch in enumerate(task_loader):

            x = batch[0].to(local_vae.device)
            y = batch[1]  # .to(local_vae.device)
            recon_x, mean, log_var, z = local_vae(x, starting_point, y, translate_noise=translate_noise)

            loss = loss_fn(recon_x, x, mean, log_var, lap_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if epoch == 0:
                class_counter = torch.unique(y, return_counts=True)
                table_tmp[class_counter[0]] += class_counter[1].cpu()
        scheduler.step()
        #     print("lr:",scheduler.get_lr())
        #     print(iteration,len(task_loader))
        if epoch % 1 == 0:
            print("Epoch: {}/{}, loss: {}, took: {} s".format(epoch, n_epochs, np.mean(losses), time.time() - start))
    return table_tmp


def train_global_decoder(curr_global_decoder, local_vae, task_id, class_table,
                         models_definition, n_sigma, n_epochs=100, n_iterations=30, batch_size=1000, train_same_z=False,
                         new_global_decoder=False):
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
    # local_vae.translator.train()
    # frozen_translator = copy.deepcopy(curr_global_decoder.translator)
    # frozen_translator.eval()
    optimizer = torch.optim.Adam(global_decoder.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    criterion = nn.MSELoss(reduction='sum')
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
                n_img=batch_size * task_id,
                return_z=True,
                translate_noise=True,
                # task_id != 1,  # Do not translate noise when generating data from 0-task in the 1st task
                same_z=train_same_z)

            if train_same_z:
                z_prev, z_max = z_prev
                z_local = z_max[:batch_size]

            with torch.no_grad():
                sampled_classes_local = class_samplers[-1].sample([batch_size])
                if not train_same_z:
                    z_local = torch.randn([batch_size, local_vae.latent_size]).to(curr_global_decoder.device)
                recon_local = local_vae.decoder(z_local, local_starting_point, sampled_classes_local,
                                                translate_noise=True)

            z_concat = torch.cat([z_prev, z_local])
            task_ids_concat = torch.cat([task_ids_prev, task_ids_local])
            class_concat = torch.cat([classes_prev, sampled_classes_local])
            if epoch > 5:
                with torch.no_grad():
                    prev_noise_translated = global_decoder.translator(z_prev, task_ids_prev)
                    current_noise_translated = global_decoder.translator(z_local, task_ids_local)
                    noise_diff_threshold = current_noise_translated.std() * n_sigma  # .5  # * 2.5  # 3

                    for prev_task_id in range(task_id):
                        selected_task_ids = torch.where(task_ids_prev == prev_task_id)[0]
                        selected_task_ids = selected_task_ids[:len(current_noise_translated)]
                        noises_distances = torch.pairwise_distance(prev_noise_translated[selected_task_ids],
                                                                   current_noise_translated[:len(selected_task_ids)])
                        if (noises_distances < noise_diff_threshold).sum() > 0:
                            imgs_task = recon_prev[selected_task_ids]
                            imgs_task[noises_distances < noise_diff_threshold] = recon_local[:len(selected_task_ids)][
                                noises_distances < noise_diff_threshold]
                            recon_prev[selected_task_ids] = imgs_task
                            # sum_changed = (noises_distances < noise_diff_threshold).sum()
                            sum_changed[prev_task_id] += (noises_distances < noise_diff_threshold).sum().item()

            recon_concat = torch.cat([recon_prev, recon_local])

            n_mini_batches = math.ceil(len(z_concat) / batch_size)
            shuffle = torch.randperm(len(task_ids_concat))
            z_concat = z_concat[shuffle]
            task_ids_concat = task_ids_concat[shuffle]
            recon_concat = recon_concat[shuffle]
            class_concat = class_concat[shuffle]

            for batch_id in range(n_mini_batches):
                start_point = batch_id * batch_size
                end_point = min(len(task_ids_concat), (batch_id + 1) * batch_size)
                global_recon = global_decoder(z_concat[start_point:end_point], task_ids_concat[start_point:end_point],
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


import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from vae_experiments.lap_loss import LapLoss
from vae_experiments.vae_utils import *
import copy

mse_loss = nn.MSELoss(reduction="sum")


def loss_fn(y, x_target, mu, sigma, lap_loss_fn=None):
    # marginal_likelihood = F.binary_cross_entropy(y, x_target, reduction='sum') / y.size(0)
    marginal_likelihood = mse_loss(y, x_target) / y.size(0)

    KL_divergence = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp()) / y.size(0)
    if lap_loss_fn:
        lap_loss = lap_loss_fn(y, x_target)
        loss = marginal_likelihood + x_target[0].size()[1] * x_target[0].size()[1] * lap_loss + KL_divergence
    else:
        loss = marginal_likelihood + KL_divergence

    return loss


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
            recon_x, _, _, _ = local_vae(x.repeat([task_id, 1, 1, 1]), task_ids, y.repeat([task_id]),
                                         translate_noise=True, noise=noise)
            for task in range(task_id):
                losses[task] += (loss(x, recon_x[task * batch_size:(task + 1) * batch_size]))
            if iteration > 10:
                break
        print(f"Starting points_losses:{losses}")
    return torch.min(losses), torch.argmin(losses).item()


def train_local_generator(local_vae, task_loader, task_id, n_classes, n_epochs=100, use_lap_loss=False):
    local_vae.train()
    local_vae.translator.train()
    # if task_id == 0:
    #     translate_noise = False
    # else:
    translate_noise = True
    min_loss, starting_point = find_best_starting_point(local_vae, task_loader, task_id)
    print(f"Selected {starting_point} as staring point for task {task_id}")
    local_vae.starting_point = starting_point
    lr = min_loss * 0.001
    print(f"lr set to: {lr}")
    scheduler_rate = 0.95
    optimizer = torch.optim.Adam(list(local_vae.parameters()) + list(local_vae.translator.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)
    table_tmp = torch.zeros(n_classes, dtype=torch.long)
    lap_loss = LapLoss(device=local_vae.device) if use_lap_loss else None
    if task_id > 0:
        optimizer = torch.optim.Adam(local_vae.encoder.parameters(), lr=lr * 5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)
    #     local_vae.decoder.eval()
    #     for name, p in local_vae.named_parameters():
    #         p.requires_grad = False

    for epoch in range(n_epochs):
        losses = []
        start = time.time()
        if epoch == 5:
            optimizer = torch.optim.Adam(list(local_vae.parameters()) + list(local_vae.translator.parameters()),
                                         lr=lr)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)
            # for name, p in local_vae.named_parameters():
            #     p.requires_grad = True
        for iteration, batch in enumerate(task_loader):

            x = batch[0].to(local_vae.device)
            y = batch[1]  # .to(local_vae.device)
            recon_x, mean, log_var, z = local_vae(x, starting_point, y, translate_noise=translate_noise)

            loss = loss_fn(recon_x, x, mean, log_var, lap_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if epoch == 0:
                class_counter = torch.unique(y, return_counts=True)
                table_tmp[class_counter[0]] += class_counter[1].cpu()
        scheduler.step()
        #     print("lr:",scheduler.get_lr())
        #     print(iteration,len(task_loader))
        if epoch % 1 == 0:
            print("Epoch: {}/{}, loss: {}, took: {} s".format(epoch, n_epochs, np.mean(losses), time.time() - start))
    return table_tmp


def train_global_decoder(curr_global_decoder, local_vae, task_id, class_table,
                         models_definition, n_sigma, n_epochs=100, n_iterations=30, batch_size=1000, train_same_z=False,
                         new_global_decoder=False):
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
    # local_vae.translator.train()
    # frozen_translator = copy.deepcopy(curr_global_decoder.translator)
    # frozen_translator.eval()
    optimizer = torch.optim.Adam(global_decoder.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    criterion = nn.MSELoss(reduction='sum')
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
                n_img=batch_size * task_id,
                return_z=True,
                translate_noise=True,
                # task_id != 1,  # Do not translate noise when generating data from 0-task in the 1st task
                same_z=train_same_z)

            if train_same_z:
                z_prev, z_max = z_prev
                z_local = z_max[:batch_size]

            with torch.no_grad():
                sampled_classes_local = class_samplers[-1].sample([batch_size])
                if not train_same_z:
                    z_local = torch.randn([batch_size, local_vae.latent_size]).to(curr_global_decoder.device)
                recon_local = local_vae.decoder(z_local, local_starting_point, sampled_classes_local,
                                                translate_noise=True)

            z_concat = torch.cat([z_prev, z_local])
            task_ids_concat = torch.cat([task_ids_prev, task_ids_local])
            class_concat = torch.cat([classes_prev, sampled_classes_local])
            if epoch > 5:
                with torch.no_grad():
                    prev_noise_translated = global_decoder.translator(z_prev, task_ids_prev)
                    current_noise_translated = global_decoder.translator(z_local, task_ids_local)
                    noise_diff_threshold = current_noise_translated.std() * n_sigma  # * 2  # .5  # * 2.5  # 3

                    for prev_task_id in range(task_id):
                        selected_task_ids = torch.where(task_ids_prev == prev_task_id)[0]
                        selected_task_ids = selected_task_ids[:len(current_noise_translated)]
                        noises_distances = torch.pairwise_distance(prev_noise_translated[selected_task_ids],
                                                                   current_noise_translated[:len(selected_task_ids)])
                        if (noises_distances < noise_diff_threshold).sum() > 0:
                            imgs_task = recon_prev[selected_task_ids]
                            imgs_task[noises_distances < noise_diff_threshold] = recon_local[:len(selected_task_ids)][
                                noises_distances < noise_diff_threshold]
                            recon_prev[selected_task_ids] = imgs_task
                            # sum_changed = (noises_distances < noise_diff_threshold).sum()
                            sum_changed[prev_task_id] += (noises_distances < noise_diff_threshold).sum().item()

            recon_concat = torch.cat([recon_prev, recon_local])

            n_mini_batches = math.ceil(len(z_concat) / batch_size)
            shuffle = torch.randperm(len(task_ids_concat))
            z_concat = z_concat[shuffle]
            task_ids_concat = task_ids_concat[shuffle]
            recon_concat = recon_concat[shuffle]
            class_concat = class_concat[shuffle]

            for batch_id in range(n_mini_batches):
                start_point = batch_id * batch_size
                end_point = min(len(task_ids_concat), (batch_id + 1) * batch_size)
                global_recon = global_decoder(z_concat[start_point:end_point], task_ids_concat[start_point:end_point],
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

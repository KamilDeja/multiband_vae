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


def train_local_generator(local_vae, task_loader, task_id, n_classes, n_epochs=100, use_lap_loss=False):
    local_vae.train()
    # if task_id == 0:
    #     translate_noise = False
    # else:
    translate_noise = False
    optimizer = torch.optim.Adam(local_vae.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    table_tmp = torch.zeros(n_classes, dtype=torch.long)
    lap_loss = LapLoss(device=local_vae.device) if use_lap_loss else None

    for epoch in range(n_epochs):
        losses = []
        for iteration, batch in enumerate(task_loader):

            x = batch[0].to(local_vae.device)
            y = batch[1]  # .to(local_vae.device)
            recon_x, mean, log_var, z = local_vae(x, task_id, y, translate_noise=translate_noise)

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
            print("Epoch: {}/{}, loss: {}".format(epoch, n_epochs, np.mean(losses)))
    return table_tmp


def train_global_decoder(curr_global_decoder, local_vae, task_id, class_table, n_epochs=100, n_iterations=30,
                         batch_size=1000, train_same_z=False):
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
    task_ids_local = torch.zeros([batch_size]) + task_id

    for epoch in range(n_epochs):
        losses = []
        for iteration in range(n_iterations):
            # Building dataset from previous global model and local model
            recon_prev, classes_prev, z_prev, task_ids_prev, embeddings_prev = generate_previous_data(
                curr_global_decoder,
                class_table=class_table,
                n_tasks=task_id,
                n_img=batch_size * task_id,
                return_z=True,
                translate_noise=task_id != 1,  # Do not translate noise when generating data from 0-task in the 1st task
                same_z=train_same_z)

            if train_same_z:
                z_prev, z_max = z_prev
                z_local = z_max[:batch_size]

            with torch.no_grad():
                sampled_classes_local = class_samplers[-1].sample([batch_size])
                if not train_same_z:
                    z_local = torch.randn([batch_size, local_vae.latent_size]).to(curr_global_decoder.device)
                recon_local = local_vae.decoder(z_local, task_ids_local, sampled_classes_local,
                                                translate_noise=False)

            z_concat = torch.cat([z_prev, z_local])
            task_ids_concat = torch.cat([task_ids_prev, task_ids_local])
            recon_concat = torch.cat([recon_prev, recon_local])

            global_recon = global_decoder(z_concat, task_ids_concat, torch.cat([classes_prev, sampled_classes_local]))
            loss = criterion(global_recon, recon_concat)
            if task_id > 1:
                new_matrix, new_bias = global_decoder.translator(task_ids_prev)  # z_prev
                prev_matrix, prev_bias = embeddings_prev
                embedding_loss = embedding_loss_criterion(new_matrix, prev_matrix)
                embedding_loss_bias = embedding_loss_criterion(new_bias, prev_bias)
                loss = loss + embedding_loss + embedding_loss_bias

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        scheduler.step()
        #     print("lr:",scheduler.get_lr())
        if (epoch % 1 == 0):
            print("Epoch: {}/{}, loss: {}".format(epoch, n_epochs, np.mean(losses)))

    # local_vae.translator = copy.deepcopy(global_decoder.translator)
    return global_decoder

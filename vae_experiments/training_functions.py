import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
# from geomloss import SamplesLoss
from torch.distributions.utils import logits_to_probs

from vae_experiments.lap_loss import LapLoss
from vae_experiments.latent_visualise import Visualizer
from vae_experiments.vae_utils import *
import copy
from vae_experiments.evaluation_models.lenet import Model

# sinkhorn_loss = SamplesLoss("sinkhorn", blur=0.05, scaling=0.95, diameter=0.01, debias=True)
torch.autograd.set_detect_anomaly(True)


def entropy(logits):
    logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = logits * logits_to_probs(logits)
    return -p_log_p.sum(-1)


def loss_fn(y, x_target, mu, sigma, marginal_loss, lap_loss_fn=None):
    # marginal_likelihood = bce_loss(y, x_target) / y.size(0)
    marginal_likelihood = marginal_loss(y, x_target) / y.size(0)
    KL_divergence = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp()) / y.size(0)
    # KL_divergence = torch.mean(KL_divergence) / y.size(1)  # / y.size(0)  # / y.size(1)
    if lap_loss_fn:
        lap_loss = lap_loss_fn(y, x_target)
        loss = marginal_likelihood + x_target[0].size()[1] * x_target[0].size()[1] * lap_loss + KL_divergence
    else:
        loss = marginal_likelihood + KL_divergence

    return loss, KL_divergence


def bin_loss_fn(x):
    utilization_loss = x.mean(0).pow(2).sum()  # (x.mean()).pow(2)  # Mean should be zero
    # x = x / 2 + 0.5
    # with torch.no_grad():
    #     targets = torch.round(x)
    # # print(x)
    # binarizatison_loss = bce_loss(x, targets) / x.size(0)
    # print(f"Utilization_loss {utilization_loss}")# Binarization_loss {binarization_loss}")
    return 0  # 50 * utilization_loss  # binarization_loss#utilization_loss + binarization_loss


def cosine_distance(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


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
            recon_x, _, _, _, _ = local_vae(x.repeat([task_id, 1, 1, 1]), task_ids, y.view(-1).repeat([task_id]),
                                            temp=1,
                                            translate_noise=True, noise=noise)
            for task in range(task_id):
                losses[task] += (loss(x, recon_x[task * batch_size:(task + 1) * batch_size])).item()
            if iteration > 10:
                break
        print(f"Starting points_losses:{losses}")
    return torch.min(losses), torch.argmin(losses).item()


def train_local_generator(local_vae, task_loader, task_id, n_classes, n_epochs=100, use_lap_loss=False,
                          local_start_lr=0.001, scale_local_lr=False):
    local_vae.train()
    local_vae.decoder.translator.train()
    translate_noise = True
    min_loss, starting_point = find_best_starting_point(local_vae, task_loader, task_id)
    starting_point = task_id
    print(f"Selected {starting_point} as staring point for task {task_id}")
    local_vae.starting_point = starting_point
    if scale_local_lr:
        lr = min_loss * local_start_lr
    else:
        lr = local_start_lr
    print(f"lr set to: {lr}")
    scheduler_rate = 0.99
    table_tmp = torch.zeros(n_classes, dtype=torch.long)
    lap_loss = LapLoss(device=local_vae.device) if use_lap_loss else None
    if local_vae.in_size == 28:
        marginal_loss = nn.BCELoss(reduction="sum")
    else:
        marginal_loss = nn.MSELoss(reduction="sum")

    if task_id > 0:
        optimizer = torch.optim.Adam(list(local_vae.encoder.parameters()), lr=lr / 10)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)
    else:
        optimizer = torch.optim.Adam(list(local_vae.parameters()), lr=lr * 10, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # scheduler_rate)

    for epoch in range(n_epochs):
        losses = []
        kl_divs = []
        start = time.time()

        if (task_id != 0) and (epoch == min(20, max(n_epochs // 10, 5))):
            print("End of local_vae pretraining")
            optimizer = torch.optim.Adam(list(local_vae.parameters()), lr=lr, weight_decay=1e-5)
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

            loss, kl_div = loss_fn(recon_x, x, mean, log_var, marginal_loss, lap_loss)
            # binary_loss = bin_loss_fn(bin_x)
            loss_final = loss  # + binary_loss
            optimizer.zero_grad()
            loss_final.backward()
            nn.utils.clip_grad_value_(local_vae.parameters(), 5.0)
            optimizer.step()

            kl_divs.append(kl_div.item())
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
            print("Epoch: {}/{}, loss: {}, kl_div: {}, took: {} s".format(epoch, n_epochs, np.mean(losses),
                                                                          np.mean(kl_divs), time.time() - start))
    if local_vae.decoder.ones_distribution == None:
        local_vae.decoder.ones_distribution = (ones_distribution.cpu().detach() / total_examples).view(1, -1)
    else:
        local_vae.decoder.ones_distribution = torch.cat([local_vae.decoder.ones_distribution,
                                                         (ones_distribution.cpu().detach() / total_examples).view(1,
                                                                                                                  -1)],
                                                        0)
    return table_tmp


def train_global_decoder(curr_global_decoder, local_vae, task_id, class_table,
                         models_definition, cosine_sim, n_epochs=100, n_iterations=30, batch_size=1000,
                         train_same_z=False,
                         new_global_decoder=False, global_lr=0.0001, limit_previous_examples=1, warmup_rounds=20,
                         num_current_to_compare=1000, experiment_name=None, visualise_latent=False):
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

    optimizer = torch.optim.Adam(list(global_decoder.parameters()), lr=global_lr)
    # optimizer = torch.optim.Adam(list(global_decoder.translator.parameters()), lr=global_lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    criterion = nn.MSELoss(reduction='sum')
    # criterion = nn.BCELoss(reduction='sum')
    class_samplers = prepare_class_samplres(task_id + 1, class_table)
    local_starting_point = torch.zeros([batch_size]) + local_vae.starting_point
    task_ids_local = torch.zeros([batch_size]) + task_id
    n_prev_examples = int(batch_size * min(task_id, 3) * limit_previous_examples)

    tmp_decoder = curr_global_decoder
    noise_diff_threshold = cosine_sim

    lenet = Model()
    model_path = "vae_experiments/evaluation_models/lenet"  # + dataset
    lenet.load_state_dict(torch.load(model_path))
    lenet.to(global_decoder.device)
    lenet.eval()

    if visualise_latent:
        visualizer = Visualizer(global_decoder, class_table, task_id=task_id, experiment_name=experiment_name)
    for epoch in range(n_epochs):
        losses = []
        start = time.time()
        sum_changed = torch.zeros([task_id + 1])
        if visualise_latent:
            visualizer.visualize_latent(global_decoder, epoch, experiment_name)
        # if epoch == warmup_rounds:
        #     # torch.save(curr_global_decoder,
        #     #            f"results/MNIST_140_test_for_analysis/model{task_id}_after_warmup_curr_decoder")
        #     optimizer = torch.optim.Adam(list(global_decoder.parameters()), lr=global_lr)  # , momentum=0.9)
        #     scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        for iteration in range(n_iterations):
            # Building dataset from previous global model and local model
            recon_prev, classes_prev, z_prev, task_ids_prev, embeddings_prev = generate_previous_data(
                tmp_decoder,
                class_table=class_table,
                n_tasks=task_id,
                n_img=n_prev_examples,
                num_local=batch_size,
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
                    z_prev, z_bin_prev = z_prev
                    z_local = torch.randn([batch_size, local_vae.latent_size]).to(curr_global_decoder.device)
                    z_bin_local = torch.bernoulli(
                        local_vae.decoder.ones_distribution[task_id].repeat([batch_size, 1])).to(
                        curr_global_decoder.device)
                    z_bin_local = z_bin_local * 2 - 1
                    # z_bin_local = torch.rand_like(z_local).to(curr_global_decoder.device)
                    # z_bin_local = torch.round(z_bin_local) * 2 - 1
                recon_local = local_vae.decoder(z_local, z_bin_local, local_starting_point, sampled_classes_local,
                                                translate_noise=True)

            z_concat = torch.cat([z_prev, z_local])
            z_bin_concat = torch.cat([z_bin_prev, z_bin_local])
            task_ids_concat = torch.cat([task_ids_prev, task_ids_local])
            class_concat = torch.cat([classes_prev, sampled_classes_local])
            # cos_sim = torch.nn.CosineSimilarity()
            if epoch > warmup_rounds:  # Warm-up epochs within which we don't switch targets
                with torch.no_grad():
                    if num_current_to_compare > 0:
                        z_current_compare = torch.randn([num_current_to_compare, global_decoder.latent_size]).to(
                            global_decoder.device)
                        bin_rand = torch.rand([num_current_to_compare, global_decoder.binary_latent_size])
                        z_bin_current_compare = (bin_rand < global_decoder.ones_distribution[task_id]).float().to(
                            global_decoder.device)
                        z_bin_current_compare = z_bin_current_compare * 2 - 1
                        task_ids_current_compare = torch.zeros(num_current_to_compare) + task_id
                    else:
                        z_current_compare = z_local
                        z_bin_current_compare = z_bin_local
                        task_ids_current_compare = task_ids_local

                    current_noise_translated = global_decoder.translator(z_current_compare, z_bin_current_compare,
                                                                         task_ids_current_compare)
                    prev_noise_translated = global_decoder.translator(z_prev, z_bin_prev, task_ids_prev)
                    noise_simmilairty = 1 - cosine_distance(prev_noise_translated,
                                                            current_noise_translated)
                    selected_examples = torch.max(noise_simmilairty, 1)[0] > noise_diff_threshold
                    if selected_examples.sum() > 0:
                        selected_replacements = torch.max(noise_simmilairty, 1)[1][selected_examples]
                        selected_z_current = z_current_compare[selected_replacements]
                        selected_z_bin_current = z_bin_current_compare[selected_replacements]
                        selected_task_ids = torch.zeros(selected_examples.sum()) + local_vae.starting_point
                        selected_new_generations = local_vae.decoder(selected_z_current, selected_z_bin_current,
                                                                     selected_task_ids, None)
                        model_scores_prev = lenet(recon_prev[selected_examples])
                        model_scores_current = lenet(selected_new_generations)
                        entropy_prev = entropy(model_scores_prev)
                        entropy_current = entropy(model_scores_current)
                        selected_examples_prev = selected_examples
                        selected_examples_prev[selected_examples_prev == True] = entropy_prev > entropy_current
                        recon_prev[selected_examples_prev] = selected_new_generations[entropy_prev > entropy_current]
                        # selected_examples_current = selected_examples
                        # selected_examples_current[selected_examples_current == True] =entropy_prev < entropy_current
                        # if num_current_to_compare == 0:
                        #     recon_local[selected_examples_current] = recon_prev[selected_examples][entropy_prev < entropy_current]

                    switches = torch.unique(task_ids_prev[selected_examples], return_counts=True)
                    for prev_task_id, sum in zip(switches[0], switches[1]):
                        sum_changed[int(prev_task_id.item())] += sum

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
                global_decoder.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(global_decoder.parameters(), 5.0)
                optimizer.step()

                losses.append(loss.item())
        scheduler.step()
        if (epoch % 1 == 0):
            print("Epoch: {}/{}, loss: {}, took: {} s".format(epoch, n_epochs, np.mean(losses), time.time() - start))
            if sum_changed.sum() > 0:
                print(
                    f"Epoch: {epoch} - changing from batches: {[(idx, n_changes) for idx, n_changes in enumerate(sum_changed.tolist())]}")
    return global_decoder

import torch.optim as optim
from vae_experiments.models_definition import *
from vae_experiments.vae_utils import *
import copy
from torch.utils.data import Dataset, DataLoader


class CodesDataset(Dataset):
    def __init__(self, full_data_loader, codes_sorted):
        batch = next(iter(full_data_loader))
        self.images = batch[0]
        self.codes = codes_sorted

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_image = self.images[idx]
        sample_codes = self.codes[idx]
        sample = sample_image, sample_codes

        return sample


def train_local_generator(local_vae, task_loader, data_loader_stable, data_loader_total, global_classes_list,
                          task_id, codes_rep, batch_size, n_epochs_pre=20, n_epochs=100):
    local_vae.train()
    optimizer = torch.optim.Adam(local_vae.parameters(), lr=0.001)
    criterion = nn.MSELoss(reduction='sum')
    for epoch in range(n_epochs_pre):
        losses = []
        losses_enc = []
        local_vae.selected_indices = []
        for iteration, batch in enumerate(task_loader):
            x = batch[0]

            x = x.to(local_vae.device)
            recon_x, loss_enc, _ = local_vae(x, task_id, codes_rep)
            recon_x = torch.sigmoid(recon_x)

            loss = criterion(recon_x, x) / batch_size
            loss_sum = loss + 10 * loss_enc * (epoch > 2)
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
            losses_enc.append(loss_enc.item())
            losses.append(loss.item())
        if epoch % 2 == 0:
            print("Epoch: {}/{}, loss: {}, loss_enc: {}".format(epoch, n_epochs_pre, np.mean(losses),
                                                                np.mean(losses_enc)))
    local_vae.selected_indices = []
    with torch.no_grad():
        for iteration, batch in enumerate(data_loader_stable):
            x = batch[0].to(local_vae.device)
            _ = local_vae(x, task_id, codes_rep)

    dataset_with_codes = CodesDataset(data_loader_total, codes_rep[0][np.array(local_vae.selected_indices)])
    if task_id > 0:
        if x.size()[3] == 32:
            batch_size = batch_size * 3
        else:
            scaling_factor = 3
            n_repeats = scaling_factor * max((5 - task_id), 1)
            batch_size = batch_size * n_repeats
    dataloader_with_codes = DataLoader(dataset_with_codes, batch_size=batch_size, shuffle=True, drop_last=True)
    global_classes_list += next(iter(data_loader_total))[1][
        np.array(local_vae.selected_indices).argsort()].cpu().detach().numpy().tolist()

    if task_id == 0:
        optimizer = torch.optim.Adam(local_vae.decoder.parameters(), lr=0.005)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        criterion = nn.MSELoss(reduction='sum')
        for epoch in range(n_epochs):
            losses = []
            local_vae.selected_indices = []
            for iteration, (x, code) in enumerate(dataloader_with_codes):
                x = x.to(local_vae.device)
                code = code.to(local_vae.device)
                task_ids = np.zeros([batch_size]) + task_id
                gen_x = local_vae.decoder.forward(code, task_ids)
                gen_x = torch.sigmoid(gen_x)
                loss = criterion(gen_x, x) / batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            scheduler.step()
            if epoch % 2 == 0:
                print("Epoch: {}/{}, loss: {}".format(epoch, n_epochs, np.mean(losses)))

    return dataloader_with_codes


def train_global_decoder(curr_global_decoder, local_vae, dataloader_with_codes, task_id, codes_rep, total_n_codes,
                         global_n_codes, global_classes_list, d, n_epochs=50, batch_size=40, n_channels=3, in_size=32):
    global_decoder = Decoder(local_vae.latent_size, d=d, p_coding=local_vae.p_coding,
                             n_dim_coding=local_vae.n_dim_coding, device=local_vae.device,
                             n_channels=n_channels, in_size=in_size).to(local_vae.device)
    curr_global_decoder.eval()
    global_decoder.train()
    optimizer = torch.optim.Adam(global_decoder.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    criterion = nn.MSELoss(reduction='sum')
    if in_size == 32:
        n_repeats = 3
    else:
        scaling_factor = 3
        n_repeats = scaling_factor * max((5 - task_id), 1)
    recreation_batch_size = n_repeats * batch_size

    starting_points = []
    for prev_task_id in range(task_id):
        starting_points.append(
            np.random.permutation(np.array(range(global_n_codes[prev_task_id] // recreation_batch_size))))
    max_len = max([len(repeats) for repeats in starting_points])
    starting_points_fixed = []
    for points in starting_points:
        starting_points_fixed.append(np.pad(points, [0, max_len - len(points)], mode="reflect"))
    starting_points_fixed = np.array(starting_points_fixed)

    for epoch in range(n_epochs):
        losses = []
        for iteration, (x, code) in enumerate(dataloader_with_codes):
            # Actual task
            x = x.to(local_vae.device)
            codes_tmp = code.to(local_vae.device)
            task_ids_local = np.zeros([len(codes_tmp)]) + task_id
            ###

            with torch.no_grad():
                recon_prev, _, task_ids_prev, codes_tmp_prev = generate_previous_data(curr_global_decoder, task_id,
                                                                                      recreation_batch_size,
                                                                                      starting_points_fixed[:,iteration] * recreation_batch_size,
                                                                                      global_classes_list,
                                                                                      total_n_codes,
                                                                                      global_n_codes,
                                                                                      return_codes=True)

                recon_local = x

                codes_concat = torch.cat([codes_tmp_prev, codes_tmp])

                task_ids_concat = np.concatenate([task_ids_prev, task_ids_local])
                recon_concat = torch.cat([recon_prev, recon_local])
                shuffle = torch.randperm(len(task_ids_concat))
                codes_concat = codes_concat[shuffle]
                recon_concat = recon_concat[shuffle]
                task_ids_concat = task_ids_concat[shuffle.detach().cpu().numpy()]

            for i in range(n_repeats):
                with torch.no_grad():
                    codes_concat_tmp = codes_concat[i * batch_size:(i + 1) * batch_size]
                    recon_concat_tmp = recon_concat[i * batch_size:(i + 1) * batch_size]
                    task_ids_concat_tmp = task_ids_concat[i * batch_size:(i + 1) * batch_size]
                optimizer.zero_grad()
                global_recon_tmp = global_decoder(codes_concat_tmp, task_ids_concat_tmp)
                global_recon_tmp = torch.sigmoid(global_recon_tmp)
                loss = criterion(global_recon_tmp, recon_concat_tmp)

                loss.backward()
                optimizer.step()

                losses.append(loss.item())
        scheduler.step()
        if epoch % 2 == 0:
            print("Epoch: {}/{}, loss: {}".format(epoch, n_epochs, np.mean(losses)))
    local_vae.decoder = copy.deepcopy(global_decoder)
    return global_decoder

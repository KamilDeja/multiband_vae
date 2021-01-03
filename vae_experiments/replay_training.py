import copy
import numpy as np
import torch
import torch.functional as F
from vae_experiments.vae_utils import generate_previous_data
from vae_experiments.training_functions import loss_fn


def train_with_replay(local_vae, task_loader, task_id, class_table, n_epochs=100):
    optimizer = torch.optim.RMSprop(local_vae.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
    freezed_model = copy.deepcopy(local_vae)

    for epoch in range(n_epochs):
        losses = []
        for iteration, (x, y) in enumerate(task_loader):
            x = x.to(local_vae.device)
            y = y.to(local_vae.device)
            if task_id > 0:
                recon_prev, recon_classes = generate_previous_data(local_vae.decoder, class_table=class_table,
                                                                   n_tasks=task_id, n_img=task_id * x.size(0))
                x = torch.cat([x, recon_prev], dim=0)
                y = torch.cat([y, recon_classes], dim=0)

            recon_x, mean, log_var, z = local_vae(x, None, y)

            loss = loss_fn(recon_x, x, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        #     scheduler.step()
        #     print("lr:",scheduler.get_lr())
        if (epoch % 10 == 0):
            print("Epoch: {}/{}, loss: {}".format(epoch, n_epochs, np.mean(losses)))

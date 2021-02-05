import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vae_experiments.vae_utils import BitUnpacker


def one_hot_conditionals(y, device, cond_dim):
    zero_ar = torch.zeros(y.shape[0], cond_dim)
    zero_ar[np.array(range(y.shape[0])), y] = 1
    return zero_ar.to(device)  # torch.Tensor(zero_ar).type(torch.float).to(device))


class VAE(nn.Module):
    def __init__(self, latent_size, d, p_coding, n_dim_coding, cond_p_coding, cond_n_dim_coding, cond_dim,
                 device, in_size):  # d defines the number of filters in conv layers of decoder and encoder
        super().__init__()
        self.p_coding = p_coding
        self.n_dim_coding = n_dim_coding
        self.latent_size = latent_size
        self.device = device

        self.encoder = Encoder(latent_size, d, cond_dim, cond_p_coding, cond_n_dim_coding, device, in_size)
        self.translator = None  # Translator(n_dim_coding, p_coding, latent_size, device)
        self.decoder = Decoder(latent_size, d, p_coding, n_dim_coding, cond_p_coding, cond_n_dim_coding, cond_dim,
                               self.translator, device, in_size)

    def forward(self, x, task_id, conds, translate_noise=True):
        batch_size = x.size(0)
        means, log_var = self.encoder(x, conds)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size]).to(self.device)
        z = eps * std + means
        if isinstance(task_id, torch.Tensor):
            task_ids = task_id
        elif task_id != None:
            task_ids = torch.zeros([batch_size, 1]) + task_id
        else:
            task_ids = torch.zeros([batch_size, 1])
        recon_x = self.decoder(z, task_ids, conds, translate_noise=translate_noise)

        return recon_x, means, log_var, z


class Encoder(nn.Module):

    def __init__(self, latent_size, d, cond_dim, cond_p_coding, cond_n_dim_coding, device, in_size):
        super().__init__()
        assert cond_dim == 10  # change cond_n_dim_coding
        self.d = d
        self.cond_p_coding = cond_p_coding
        self.cond_n_dim_coding = cond_n_dim_coding
        self.cond_dim = cond_dim
        self.device = device
        self.in_size = in_size
        if self.in_size == 28:
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.d, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_1 = nn.BatchNorm2d(self.d)
            self.conv2 = nn.Conv2d(self.d, self.d, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_2 = nn.BatchNorm2d(self.d)
            self.conv3 = nn.Conv2d(self.d, self.d, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_3 = nn.BatchNorm2d(self.d)
            # self.fc3 = nn.Linear(self.d * 9, self.d)
            self.fc = nn.Linear(self.d * 9 + cond_n_dim_coding, self.d * 4)

        else:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.d, kernel_size=5, stride=2, padding=1, bias=False)
            self.bn_1 = nn.BatchNorm2d(self.d)

            self.conv2 = nn.Conv2d(self.d, self.d * 2, kernel_size=5, stride=2, padding=1, bias=False)
            self.bn_2 = nn.BatchNorm2d(self.d * 2)

            self.conv3 = nn.Conv2d(self.d * 2, self.d * 4, kernel_size=5, stride=2, padding=1, bias=False)
            self.bn_3 = nn.BatchNorm2d(self.d * 4)

            self.conv4 = nn.Conv2d(self.d * 4, self.d * 4, kernel_size=5, stride=2, padding=1, bias=False)
            self.bn_4 = nn.BatchNorm2d(self.d * 4)
            self.fc = nn.Linear(self.d * 4 * 3 * 3 + cond_n_dim_coding, self.d * 4)

        self.linear_means = nn.Linear(self.d * 4, latent_size)
        self.linear_log_var = nn.Linear(self.d * 4, latent_size)

    def forward(self, x, conds):
        with torch.no_grad():
            if self.cond_n_dim_coding:
                conds_coded = (conds * self.cond_p_coding) % (2 ** self.cond_n_dim_coding)
                conds_coded = BitUnpacker.unpackbits(conds_coded, self.cond_n_dim_coding).to(self.device)

        x = self.conv1(x)
        x = F.leaky_relu(self.bn_1(x))
        x = self.conv2(x)
        x = F.leaky_relu(self.bn_2(x))
        x = self.conv3(x)
        x = F.leaky_relu(self.bn_3(x))
        if self.in_size == 28:
            x = x.view([-1, self.d * 9])
        else:
            x = self.conv4(x)
            x = F.leaky_relu(self.bn_4(x))
            x = x.view([-1, self.d * 4 * 3 * 3])

        if self.cond_n_dim_coding:
            x = torch.cat([x, conds_coded], dim=1)

        x = F.leaky_relu(self.fc(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, latent_size, d, p_coding, n_dim_coding, cond_p_coding, cond_n_dim_coding, cond_dim, translator,
                 device, in_size):
        super().__init__()
        self.d = d
        self.p_coding = p_coding
        self.n_dim_coding = n_dim_coding
        self.cond_p_coding = cond_p_coding
        self.cond_n_dim_coding = cond_n_dim_coding
        self.cond_dim = cond_dim
        self.device = device
        self.latent_size = latent_size
        self.translator = translator
        self.bias_mockup = torch.Tensor([0.5] * self.n_dim_coding).to(self.device)

        # self.fc0 = nn.Linear(latent_size, latent_size)
        self.fc1 = nn.Linear(latent_size + cond_n_dim_coding + n_dim_coding, self.d * 4)
        if in_size == 28:
            self.scaler = 4
            self.fc2 = nn.Linear(self.d * 4, self.d * 8)
            self.fc3 = nn.Linear(self.d * 8, self.d * self.scaler * self.scaler * self.scaler)
            self.dc1 = nn.ConvTranspose2d(self.d * self.scaler, self.d * self.scaler, kernel_size=4, stride=2,
                                          padding=0, bias=False)
            self.dc1_bn = nn.BatchNorm2d(self.d * 4)
            self.dc2 = nn.ConvTranspose2d(self.d * 4, self.d * 2, kernel_size=4, stride=2, padding=0, bias=False)
            self.dc2_bn = nn.BatchNorm2d(self.d * 2)
            self.dc3 = nn.ConvTranspose2d(self.d * 2, self.d, kernel_size=4, stride=1, padding=0, bias=False)
            self.dc3_bn = nn.BatchNorm2d(self.d)
            self.dc_out = nn.ConvTranspose2d(self.d, 1, kernel_size=4, stride=1, padding=0, bias=False)
        else:
            self.scaler = 8
            self.fc2 = nn.Linear(self.d * 4, self.d * 8)
            self.fc3 = nn.Linear(self.d * 8, self.d * 8 * 8 * 8)
            self.dc1 = nn.ConvTranspose2d(self.d * 8, self.d * 4, kernel_size=5, stride=2,
                                          padding=2, output_padding=1, bias=False)
            self.dc1_bn = nn.BatchNorm2d(self.d * 4)

            self.dc2 = nn.ConvTranspose2d(self.d * 4, self.d * 2, kernel_size=5, stride=2,
                                          padding=2, output_padding=1, bias=False)
            self.dc2_bn = nn.BatchNorm2d(self.d * 2)

            self.dc3 = nn.ConvTranspose2d(self.d * 2, self.d, kernel_size=5, stride=2,
                                          padding=2, output_padding=1, bias=False)
            self.dc3_bn = nn.BatchNorm2d(self.d)

            self.dc_out = nn.ConvTranspose2d(self.d, 3, kernel_size=5, stride=1,
                                             padding=2, output_padding=0, bias=False)

    def forward(self, x, task_id, conds, return_emb=False, translate_noise=True):
        with torch.no_grad():
            if self.cond_n_dim_coding:
                conds_coded = (conds * self.cond_p_coding) % (2 ** self.cond_n_dim_coding)
                conds_coded = BitUnpacker.unpackbits(conds_coded, self.cond_n_dim_coding).to(self.device)

                x = torch.cat([x, conds_coded], dim=1)

            batch_conds_coded = (task_id * self.p_coding) % (2 ** self.n_dim_coding)
            batch_conds_coded = BitUnpacker.unpackbits(batch_conds_coded, self.n_dim_coding).to(self.device)

            x = torch.cat([x, batch_conds_coded], dim=1)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = x.view(-1, self.d * self.scaler, self.scaler, self.scaler)
        x = self.dc1(x)
        x = F.leaky_relu(self.dc1_bn(x))
        x = self.dc2(x)
        x = F.leaky_relu(self.dc2_bn(x))
        x = self.dc3(x)
        x = F.leaky_relu(self.dc3_bn(x))
        x = torch.sigmoid(self.dc_out(x))
        if return_emb:
            return x, (None, None)
        return x
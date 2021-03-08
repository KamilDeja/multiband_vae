from collections import OrderedDict

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
    def __init__(self, latent_size, binary_latent_size, d, p_coding, n_dim_coding, cond_p_coding, cond_n_dim_coding,
                 cond_dim, device, in_size, fc, standard_embeddings=False,
                 trainable_embeddings=False):  # d defines the number of filters in conv layers of decoder and encoder
        super().__init__()
        self.p_coding = p_coding
        self.n_dim_coding = n_dim_coding
        self.latent_size = latent_size
        self.binary_latent_size = binary_latent_size
        self.device = device
        self.standard_embeddings = standard_embeddings
        self.in_size = in_size
        self.starting_point = None
        self.temp = 0.001

        self.encoder = Encoder(latent_size, binary_latent_size, d, cond_dim, cond_p_coding, cond_n_dim_coding, device,
                               in_size, fc)
        if standard_embeddings:
            translator = Translator_embeddings(n_dim_coding, p_coding, latent_size, device)
        else:
            translator = Translator(n_dim_coding, p_coding, latent_size, binary_latent_size, device, d=d)
        self.decoder = Decoder(latent_size, binary_latent_size, d, p_coding, n_dim_coding, cond_p_coding,
                               cond_n_dim_coding, cond_dim,
                               translator, device, standard_embeddings=standard_embeddings,
                               trainable_embeddings=trainable_embeddings, in_size=in_size, fc=fc)

    def forward(self, x, task_id, conds, temp, translate_noise=True, noise=None, encode_to_noise=False):
        batch_size = x.size(0)
        if temp == None:
            hard = True
            temp = 1
        else:
            hard = False
        means, log_var, binary_out = self.encoder(x, conds)
        std = torch.exp(0.5 * log_var)
        binary_out_reverse = - binary_out
        if self.binary_latent_size > 0:
            binary_out_merged = torch.stack([binary_out, binary_out_reverse], -1)
            binary_out = F.gumbel_softmax(binary_out_merged, tau=temp, hard=hard, dim=2)[:, :, 0]
        binary_out = binary_out * 2 - 1
        # print(binary_out)

        if noise == None:
            eps = torch.randn([batch_size, self.latent_size]).to(self.device)
        else:
            eps = noise
        z = eps * std + means
        if not torch.is_tensor(task_id):
            if task_id != None:
                task_id = torch.zeros([batch_size, 1]) + task_id
            else:
                task_id = torch.zeros([batch_size, 1])
        if encode_to_noise:
            emb = self.decoder.translator(z, binary_out, task_id)
            return emb
        recon_x = self.decoder(z, binary_out, task_id, conds, translate_noise=translate_noise)

        return recon_x, means, log_var, z, binary_out


class Encoder(nn.Module):

    def __init__(self, latent_size, binary_latent_size, d, cond_dim, cond_p_coding, cond_n_dim_coding, device, in_size,
                 fc):
        super().__init__()
        # assert cond_dim == 10  # change cond_n_dim_coding
        assert cond_n_dim_coding == 0  # Doesn't work right now
        self.d = d
        self.cond_p_coding = cond_p_coding
        self.cond_n_dim_coding = cond_n_dim_coding
        self.cond_dim = cond_dim
        self.device = device
        self.in_size = in_size
        self.only_fc = fc
        if self.in_size == 28:
            in_channels = 1
            self.scaler = 4
        else:
            in_channels = 3
            self.scaler = 8
        if self.only_fc:
            self.fc_1 = nn.Linear(in_size * in_size * in_channels + cond_n_dim_coding,
                                  self.d * self.scaler * self.scaler)
            self.fc_2 = nn.Linear(self.d * self.scaler * self.scaler, self.d * self.scaler)
            self.fc_3 = nn.Linear(self.d * self.scaler, self.d * self.scaler // 2)
            self.linear_means = nn.Linear(self.d * self.scaler // 2, latent_size)
            self.linear_log_var = nn.Linear(self.d * self.scaler // 2, latent_size)
            self.linear_binary = nn.Linear(self.d * self.scaler // 2, binary_latent_size)
        else:
            if self.in_size == 28:
                self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.d, kernel_size=4, stride=2, padding=1,
                                       bias=False)
                self.bn_1 = nn.BatchNorm2d(self.d)
                self.conv2 = nn.Conv2d(self.d, self.d, kernel_size=4, stride=2, padding=1, bias=False)
                self.bn_2 = nn.BatchNorm2d(self.d)
                self.conv3 = nn.Conv2d(self.d, self.d, kernel_size=4, stride=2, padding=1, bias=False)
                self.bn_3 = nn.BatchNorm2d(self.d)
                # self.fc3 = nn.Linear(self.d * 9, self.d)
                self.fc = nn.Linear(self.d * 9 + cond_n_dim_coding, self.d * 4)

            else:
                self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.d, kernel_size=5, stride=2, padding=1,
                                       bias=False)
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
            self.linear_binary = nn.Linear(self.d * 4, binary_latent_size, bias=False)

    def forward(self, x, conds):
        with torch.no_grad():
            if self.cond_n_dim_coding:
                conds_coded = (conds * self.cond_p_coding) % (2 ** self.cond_n_dim_coding)
                conds_coded = BitUnpacker.unpackbits(conds_coded, self.cond_n_dim_coding).to(self.device)
                x = torch.cat([x, conds_coded], dim=1)
        if self.only_fc:
            x = x.view(x.size(0), -1)
            x = F.leaky_relu(self.fc_1(x))
            x = F.leaky_relu(self.fc_2(x))
            x = F.leaky_relu(self.fc_3(x))
        else:
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
        binary_out = self.linear_binary(x)
        return means, log_vars, binary_out


class Decoder(nn.Module):
    def __init__(self, latent_size, binary_latent_size, d, p_coding, n_dim_coding, cond_p_coding, cond_n_dim_coding,
                 cond_dim, translator, device, standard_embeddings, trainable_embeddings, in_size, fc):
        super().__init__()
        self.d = d
        self.p_coding = p_coding
        self.n_dim_coding = n_dim_coding
        self.cond_p_coding = cond_p_coding
        self.cond_n_dim_coding = cond_n_dim_coding
        self.cond_dim = cond_dim
        self.device = device
        self.latent_size = latent_size
        self.binary_latent_size = binary_latent_size
        self.translator = translator
        self.standard_embeddings = standard_embeddings
        self.trainable_embeddings = trainable_embeddings
        self.in_size = in_size
        self.fc = fc
        self.ones_distribution = None  # torch.zeros(binary_latent_size)
        # self.fc0 = nn.Linear(latent_size, latent_size)

        if in_size == 28:
            self.scaler = 4
            out_channels = 1
        else:
            self.scaler = 8
            out_channels = 3

        if fc:
            if self.standard_embeddings:
                self.fc1 = nn.Linear(latent_size * self.d + cond_n_dim_coding + n_dim_coding, self.d * self.scaler)
            else:
                self.fc1 = nn.Linear(latent_size * self.d + cond_n_dim_coding, self.d * self.scaler * self.scaler)
            self.fc_2 = nn.Linear(self.d * self.scaler * self.scaler, self.d * self.scaler * self.scaler * 2)
            self.fc_out = nn.Linear(self.d * self.scaler * self.scaler * 2, in_size * in_size * out_channels)
        else:
            if self.standard_embeddings:
                self.fc1 = nn.Linear(latent_size * self.d + cond_n_dim_coding + n_dim_coding,
                                     self.d * self.scaler * self.scaler * self.scaler)
            else:
                self.fc1 = nn.Linear(latent_size * self.d + cond_n_dim_coding,
                                     self.d * self.scaler * self.scaler * self.scaler)
            if in_size == 28:
                self.scaler = 4
                # self.fc2 = nn.Linear(self.d * 4, self.d * 8)
                # self.fc3 = nn.Linear(latent_size + cond_n_dim_coding, self.d * self.scaler * self.scaler * self.scaler)
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
                # self.fc2 = nn.Linear(self.d * 4, self.d * 8)
                # self.fc3 = nn.Linear(latent_size + cond_n_dim_coding, self.d * 8 * 8 * 8)
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

    def forward(self, x, binary_x, task_id, conds, return_emb=False, translate_noise=True):
        with torch.no_grad():
            if self.cond_n_dim_coding:
                conds_coded = (conds * self.cond_p_coding) % (2 ** self.cond_n_dim_coding)
                conds_coded = BitUnpacker.unpackbits(conds_coded, self.cond_n_dim_coding).to(self.device)

        if self.standard_embeddings:
            task_ids_enc = self.translator(task_id, self.trainable_embeddings)
            x = torch.cat([x, task_ids_enc], dim=1)
        elif translate_noise:
            x = self.translator(x, binary_x, task_id)
            translator_out = x

        if self.cond_n_dim_coding:
            x = torch.cat([x, conds_coded], dim=1)

        if self.fc:
            x = F.leaky_relu(self.fc1(x))
            x = F.leaky_relu(self.fc_2(x))
            x = torch.sigmoid(self.fc_out(x))
            x = x.view(x.size(0), 1, self.in_size, self.in_size)
        else:
            x = self.fc1(x)
            x = x.view(-1, self.d * self.scaler, self.scaler, self.scaler)
            x = self.dc1(x)
            x = F.leaky_relu(self.dc1_bn(x))
            x = self.dc2(x)
            x = F.leaky_relu(self.dc2_bn(x))
            x = self.dc3(x)
            x = F.leaky_relu(self.dc3_bn(x))
            x = torch.sigmoid(self.dc_out(x))
        if return_emb:
            return x, translator_out
        return x


class Translator(nn.Module):
    def __init__(self, n_dim_coding, p_coding, latent_size, binary_latent_size, device, d):
        super().__init__()
        self.n_dim_coding = n_dim_coding
        self.p_coding = p_coding
        self.device = device
        self.latent_size = latent_size
        self.binary_latent_size = binary_latent_size
        self.d = d

        self.fc_enc_1 = nn.Linear(n_dim_coding, n_dim_coding * 3)
        self.fc_enc_2 = nn.Linear(n_dim_coding * 3, n_dim_coding * 2)
        if binary_latent_size > 0:
            self.fc_bin_enc_1 = nn.Linear(binary_latent_size, binary_latent_size * 2)
            self.fc_bin_enc_2 = nn.Linear(binary_latent_size * 2, binary_latent_size * 3)
            # self.fc_enc_joined = nn.Linear(binary_latent_size * 3 + n_dim_coding * 3,
            #                                binary_latent_size * 3 + n_dim_coding * 3)
        self.fc1 = nn.Linear(n_dim_coding * 2 + latent_size + binary_latent_size * 3, latent_size * self.d // 2)
        # self.fc2 = nn.Linear(latent_size * self.d // 2, max(latent_size * n_dim_coding, 32))
        # self.fc3 = nn.Linear(max(latent_size * n_dim_coding, 32), latent_size * latent_size)
        # self.fc4 = nn.Linear(max(latent_size * n_dim_coding, 32), latent_size)
        self.fc4 = nn.Linear(latent_size * self.d // 2, latent_size * self.d)

    def forward(self, x, bin_x, task_id):
        codes = (task_id * self.p_coding) % (2 ** self.n_dim_coding)
        task_ids = BitUnpacker.unpackbits(codes, self.n_dim_coding).to(self.device)
        task_ids = F.leaky_relu(self.fc_enc_1(task_ids))
        task_ids = F.leaky_relu(self.fc_enc_2(task_ids))
        if self.binary_latent_size > 0:
            bin_x = F.leaky_relu(self.fc_bin_enc_1(bin_x))
            bin_x = F.leaky_relu(self.fc_bin_enc_2(bin_x))
        #     enc = torch.cat([bin_x, task_ids], dim=1)
        #     enc = F.leaky_relu(self.fc_enc_joined(enc))
        #     x = torch.cat([x, enc], dim=1)
        # else:
        # xi = x[:, :4]
        # xi = F.leaky_relu(self.fc_new_bin_1(xi))
        # xi = self.fc_new_bin_2(xi)
        # xi = torch.sign(xi) * torch.abs(xi).pow(1 / 3) * torch.exp(-0.5 * xi.pow(2))
        x = torch.cat([x, bin_x, task_ids], dim=1)
        x = F.leaky_relu(self.fc1(x))
        out = self.fc4(x)
        return out  # task_ids_enc_resized, bias


class Translator_embeddings(nn.Module):
    def __init__(self, n_dim_coding, p_coding, latent_size, device):
        super().__init__()
        self.n_dim_coding = n_dim_coding
        self.p_coding = p_coding
        self.device = device
        self.latent_size = latent_size

        self.fc1 = nn.Linear(n_dim_coding, n_dim_coding * 2)
        self.fc2 = nn.Linear(n_dim_coding * 2, n_dim_coding * 3)
        self.fc3 = nn.Linear(n_dim_coding * 3, n_dim_coding)

    def forward(self, task_id, trainable_embeddings):
        codes = (task_id * self.p_coding) % (2 ** self.n_dim_coding)
        task_ids = BitUnpacker.unpackbits(codes, self.n_dim_coding).to(self.device)
        # return task_ids
        if not trainable_embeddings:
            return task_ids
        x = F.leaky_relu(self.fc1(task_ids))
        x = F.leaky_relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

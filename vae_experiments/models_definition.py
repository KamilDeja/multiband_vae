import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def unpackbits(x, num_bits):
    x = x.reshape([-1, 1])
    mask = 2 ** (num_bits - 1 - np.arange(num_bits).reshape([1, num_bits]))
    return (x & mask).astype(bool).astype(int)


big_number = 99999


def one_hot_conditionals(y, device, cond_dim):
    zero_ar = torch.zeros(y.shape[0], cond_dim)
    zero_ar[np.array(range(y.shape[0])), y] = 1
    return zero_ar.to(device)


class VAE(nn.Module):
    def __init__(self, latent_size, d, p_coding, n_dim_coding, device, n_channels=3,
                 in_size=32):  # d defines the number of filters in conv layers of decoder and encoder
        super().__init__()
        self.p_coding = p_coding
        self.n_dim_coding = n_dim_coding
        self.latent_size = latent_size
        self.device = device
        self.selected_indices = []

        self.encoder = Encoder(latent_size, d, device, n_channels, in_size)
        self.decoder = Decoder(latent_size, d, p_coding, n_dim_coding, device, n_channels, in_size)

    def forward(self, x, task_id, codes_rep):
        batch_size = x.size(0)
        embeddings = self.encoder(x)

        embeddings_rep = embeddings.repeat([len(codes_rep[0]), 1, 1]).transpose(0, 1)
        if batch_size != codes_rep.shape[0]:
            codes_rep = codes_rep[0].repeat([batch_size, 1, 1])
        min_codes = torch.pow(codes_rep - embeddings_rep, 2).sum(2)
        sum_dist = 0
        min_codes[:, self.selected_indices] = big_number
        selected_indices = []

        for i in range(batch_size):
            local_min = min_codes[i].min(0)
            sum_dist += local_min.values
            selected_indices.append(int(local_min.indices))
            min_codes[:, local_min.indices] = big_number

        self.selected_indices += selected_indices

        task_ids = np.zeros([batch_size]) + task_id
        recon_x = self.decoder(embeddings, task_ids)

        return recon_x, sum_dist, selected_indices


class Encoder(nn.Module):
    def __init__(self, latent_size, d, device, n_channels, in_size):
        super().__init__()
        self.d = d
        self.device = device
        self.conv_out_dim = int(np.ceil(in_size / 8))
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=self.d, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn_1 = nn.BatchNorm2d(self.d)
        self.conv2 = nn.Conv2d(self.d, self.d * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(self.d * 2)
        self.conv3 = nn.Conv2d(self.d * 2, self.d * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(self.d * 4)
        self.out = nn.Linear(self.d * 4 * self.conv_out_dim * self.conv_out_dim, latent_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(self.bn_1(x))
        x = self.conv2(x)
        x = F.leaky_relu(self.bn_2(x))
        x = self.conv3(x)
        x = F.leaky_relu(self.bn_3(x))
        x = x.view([-1, self.d * 4 * self.conv_out_dim * self.conv_out_dim])
        encodings = self.out(x)

        return encodings


class Decoder(nn.Module):
    def __init__(self, latent_size, d, p_coding, n_dim_coding, device, n_channels, in_size):
        super().__init__()
        self.d = d
        self.p_coding = p_coding
        self.n_dim_coding = n_dim_coding
        self.device = device
        self.latent_size = latent_size

        self.fc1 = nn.Linear(latent_size + n_dim_coding, int(latent_size*1.5))
        self.fc2 = nn.Linear(int(latent_size*1.5), self.d * 64)
        self.dc1 = nn.ConvTranspose2d(self.d, self.d * 4, kernel_size=3, stride=2, output_padding=0, padding=2,
                                      bias=False)
        self.dc1_bn = nn.BatchNorm2d(self.d * 4)
        if in_size == 32:
            self.dc2 = nn.ConvTranspose2d(self.d * 4, self.d * 6, kernel_size=4, stride=2, output_padding=0, bias=False)
            self.dc2_bn = nn.BatchNorm2d(self.d * 6)
            self.dc3 = nn.ConvTranspose2d(self.d * 6, self.d * 5, kernel_size=4, stride=1, output_padding=0, bias=False)
            self.dc3_bn = nn.BatchNorm2d(self.d * 5)
            self.dc4 = nn.ConvTranspose2d(self.d * 5, n_channels, kernel_size=4, stride=1, padding=1, bias=False)
        else:
            self.dc2 = nn.ConvTranspose2d(self.d * 4, self.d * 6, kernel_size=3, stride=2, output_padding=0, bias=False)
            self.dc2_bn = nn.BatchNorm2d(self.d * 6)
            self.dc3 = nn.ConvTranspose2d(self.d * 6, self.d * 5, kernel_size=3, stride=1, output_padding=0, bias=False)
            self.dc3_bn = nn.BatchNorm2d(self.d * 5)
            self.dc4 = nn.ConvTranspose2d(self.d * 5, n_channels, kernel_size=4, stride=1, padding=2, bias=False)

    def forward(self, x, task_id):
        codes = np.array(task_id * self.p_coding % 2 ** self.n_dim_coding, dtype=np.long)
        task_ids = torch.from_numpy(unpackbits(codes, self.n_dim_coding).astype(np.float32)).to(self.device)

        x = torch.cat([x, task_ids], axis=1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = x.view(-1, self.d, 8, 8)
        x = self.dc1(x)
        x = F.leaky_relu(self.dc1_bn(x))
        x = self.dc2(x)
        x = F.leaky_relu(self.dc2_bn(x))
        x = self.dc3(x)
        x = F.leaky_relu(self.dc3_bn(x))

        return self.dc4(x)

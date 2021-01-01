import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def one_hot_conditionals(y, device, cond_dim):
    zero_ar = torch.zeros(y.shape[0], cond_dim)
    zero_ar[np.array(range(y.shape[0])), y] = 1
    return (zero_ar.to(device))  # torch.Tensor(zero_ar).type(torch.float).to(device))

def unpackbits(x, num_bits):
    # xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2 ** (num_bits - 1 - np.arange(num_bits).reshape([1, num_bits]))
    return (x & mask).astype(bool).astype(int)


class VAE(nn.Module):
    def __init__(self, latent_size, d, p_coding, n_dim_coding, cond_dim,
                 device):  # d defines the number of filters in conv layers of decoder and encoder
        super().__init__()
        self.p_coding = p_coding
        self.n_dim_coding = n_dim_coding
        self.latent_size = latent_size
        self.device = device

        self.encoder = Encoder(latent_size, d, cond_dim, device)
        self.decoder = Decoder(latent_size, d, p_coding, n_dim_coding, cond_dim, device)

    def forward(self, x, task_id, conds):
        batch_size = x.size(0)
        means, log_var = self.encoder(x, conds)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size]).to(self.device)
        z = eps * std + means

        task_ids = np.zeros([batch_size, 1]) + task_id
        recon_x = self.decoder(z, task_ids, conds)

        return recon_x, means, log_var, z

    def inference(self, n=1, task_id=0):
        batch_size = n
        z = torch.randn([batch_size, self.latent_size]).to(self.device)
        recon_x = self.decoder(z, task_id)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, latent_size, d, cond_dim, device):
        super().__init__()
        self.d = d
        self.cond_dim = cond_dim
        self.device = device
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.d, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(self.d)
        self.conv2 = nn.Conv2d(self.d, self.d, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(self.d)
        self.conv3 = nn.Conv2d(self.d, self.d, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(self.d)
        self.fc3 = nn.Linear(self.d * 9, self.d)
        self.fc4 = nn.Linear(self.d + cond_dim, self.d)

        self.linear_means = nn.Linear(self.d, latent_size)
        self.linear_log_var = nn.Linear(self.d, latent_size)

    def forward(self, x, conds):
        conds = one_hot_conditionals(conds, self.device, self.cond_dim)
        x = self.conv1(x)
        x = F.leaky_relu(self.bn_1(x))
        x = self.conv2(x)
        x = F.leaky_relu(self.bn_2(x))
        x = self.conv3(x)
        x = F.leaky_relu(self.bn_3(x))
        x = x.view([-1, self.d * 9])
        x = F.leaky_relu(self.fc3(x))
        x = torch.cat([x, conds], axis=1)
        x = self.fc4(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, latent_size, d, p_coding, n_dim_coding, cond_dim, device):
        super().__init__()
        self.d = d
        self.p_coding = p_coding
        self.n_dim_coding = n_dim_coding
        self.cond_dim = cond_dim
        self.device = device
        self.latent_size = latent_size

        self.fc1 = nn.Linear(latent_size + n_dim_coding + cond_dim,
                             self.d * 4)
        self.dc1 = nn.ConvTranspose2d(self.d, self.d * 4, 4, 2, 0, bias=False)
        self.dc1_bn = nn.BatchNorm2d(self.d * 4)
        self.dc2 = nn.ConvTranspose2d(self.d * 4, self.d * 2, 4, 2, 1, bias=False)
        self.dc2_bn = nn.BatchNorm2d(self.d * 2)
        self.dc3 = nn.ConvTranspose2d(self.d * 2, self.d, 4, 1, 1, bias=False)
        self.dc3_bn = nn.BatchNorm2d(self.d)
        self.dc4 = nn.ConvTranspose2d(self.d, 1, 4, 2, 0, bias=False)

    def forward(self, x, task_id, conds):
        codes = task_id * self.p_coding % 2 ** self.n_dim_coding
        task_ids = torch.from_numpy(
            np.unpackbits(codes.astype(np.uint8).reshape(-1, 1), axis=1)[:, -self.n_dim_coding:].astype(np.float32)).to(
            self.device)
        conds_coded = one_hot_conditionals(conds, self.device, self.cond_dim)
        x = torch.cat([x, task_ids, conds_coded], axis=1)
        x = F.leaky_relu(self.fc1(x))
        x = x.view(-1, self.d, 2, 2)
        x = self.dc1(x)
        x = F.leaky_relu(self.dc1_bn(x))
        x = self.dc2(x)
        x = F.leaky_relu(self.dc2_bn(x))
        x = self.dc3(x)
        x = F.leaky_relu(self.dc3_bn(x))
        return torch.sigmoid(self.dc4(x))

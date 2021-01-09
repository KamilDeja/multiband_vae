import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def one_hot_conditionals(y, device, cond_dim):
    zero_ar = torch.zeros(y.shape[0], cond_dim)
    zero_ar[np.array(range(y.shape[0])), y] = 1
    return zero_ar.to(device)  # torch.Tensor(zero_ar).type(torch.float).to(device))


def unpackbits(x, num_bits):
    with torch.no_grad():
        if num_bits == 0:
            return torch.Tensor([])
        x = x.view(-1, 1).long()
        mask = 2 ** (num_bits - 1 - torch.arange(num_bits).view([1, num_bits])).long()
        return (x & mask).bool().float()


class VAE(nn.Module):
    def __init__(self, latent_size, d, p_coding, n_dim_coding, cond_p_coding, cond_n_dim_coding, cond_dim,
                 device):  # d defines the number of filters in conv layers of decoder and encoder
        super().__init__()
        self.p_coding = p_coding
        self.n_dim_coding = n_dim_coding
        self.latent_size = latent_size
        self.device = device

        self.encoder = Encoder(latent_size, d, cond_dim, cond_p_coding, cond_n_dim_coding, device)
        self.translator = Translator(n_dim_coding, p_coding, latent_size, device)
        self.decoder = Decoder(latent_size, d, p_coding, n_dim_coding, cond_p_coding, cond_n_dim_coding, cond_dim,
                               self.translator, device)

    def forward(self, x, task_id, conds, translate_noise=True):
        batch_size = x.size(0)
        means, log_var = self.encoder(x, conds)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size]).to(self.device)
        z = eps * std + means
        if task_id != None:
            task_ids = torch.zeros([batch_size, 1]) + task_id
        else:
            task_ids = torch.zeros([batch_size, 1])
        recon_x = self.decoder(z, task_ids, conds, translate_noise=translate_noise)

        return recon_x, means, log_var, z


class Encoder(nn.Module):

    def __init__(self, latent_size, d, cond_dim, cond_p_coding, cond_n_dim_coding, device):
        super().__init__()
        assert cond_dim == 10  # change cond_n_dim_coding
        self.d = d
        self.cond_p_coding = cond_p_coding
        self.cond_n_dim_coding = cond_n_dim_coding
        self.cond_dim = cond_dim
        self.device = device
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
            conds_coded = (conds * self.cond_p_coding) % (2 ** self.cond_n_dim_coding)
            conds_coded = unpackbits(conds_coded, self.cond_n_dim_coding).to(self.device)
        # conds = one_hot_conditionals(conds, self.device, self.cond_dim)
        x = self.conv1(x)
        x = F.leaky_relu(self.bn_1(x))
        x = self.conv2(x)
        x = F.leaky_relu(self.bn_2(x))
        x = self.conv3(x)
        x = F.leaky_relu(self.bn_3(x))
        x = self.conv4(x)
        x = F.leaky_relu(self.bn_4(x))
        # x = x.view([-1, self.d * 9])
        # x = F.leaky_relu(self.fc3(x))
        x = x.view([-1, self.d * 4 * 3 * 3])
        x = torch.cat([x, conds_coded], dim=1)
        x = F.leaky_relu(self.fc(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, latent_size, d, p_coding, n_dim_coding, cond_p_coding, cond_n_dim_coding, cond_dim, translator,
                 device):
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

        self.fc1 = nn.Linear(latent_size + cond_n_dim_coding + n_dim_coding, self.d * 4)
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
            # torch.from_numpy(
            # np.unpackbits(codes.astype(np.uint8).reshape(-1, 1), axis=1)[:, -self.n_dim_coding:].astype(np.float32)).to(
            # self.device)
            conds_coded = (conds * self.cond_p_coding) % (2 ** self.cond_n_dim_coding)
            conds_coded = unpackbits(conds_coded, self.cond_n_dim_coding).to(
                self.device)  # one_hot_conditionals(conds, self.device, self.cond_dim)
        if translate_noise:
            # task_id = torch.cat([x, task_id.to(self.device)], dim=1)
            task_ids_enc_resized, bias = self.translator(task_id)
            x = torch.bmm(task_ids_enc_resized, x.unsqueeze(-1)).squeeze(2)
        else:
            task_ids_enc_resized = None
            bias = self.bias_mockup.repeat([x.size(0), 1])
        # x = torch.cat([x, conds_coded], dim=1)
        # task_ids_enc = self.translator(task_id)
        x = torch.cat([x, conds_coded, bias], dim=1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = x.view(-1, self.d * 8, 8, 8)

        #         print(x.size())
        #         print(x.size())
        x = self.dc1(x)
        x = F.leaky_relu(self.dc1_bn(x))
        #         print(x.size())
        x = self.dc2(x)
        x = F.leaky_relu(self.dc2_bn(x))
        #         print(x.size())
        x = self.dc3(x)
        x = F.leaky_relu(self.dc3_bn(x))
        #         print(x.size())
        #         x = self.dc5(x)
        #         x = F.leaky_relu(self.dc5_bn(x))
        x = torch.sigmoid(self.dc_out(x))
        if return_emb:
            return x, (task_ids_enc_resized, bias)
        return x


class Translator(nn.Module):
    def __init__(self, n_dim_coding, p_coding, latent_size, device):
        super().__init__()
        self.n_dim_coding = n_dim_coding
        self.p_coding = p_coding
        self.device = device
        self.latent_size = latent_size

        self.fc1 = nn.Linear(n_dim_coding, latent_size)
        self.fc2 = nn.Linear(latent_size, latent_size * n_dim_coding)
        self.fc3 = nn.Linear(latent_size * n_dim_coding, latent_size * latent_size)
        self.fc4 = nn.Linear(latent_size * n_dim_coding, n_dim_coding)

    def forward(self, task_id):
        codes = (task_id * self.p_coding) % (2 ** self.n_dim_coding)
        task_ids = unpackbits(codes, self.n_dim_coding).to(self.device)
        # x = torch.cat([x, task_ids], dim=1)
        x = F.leaky_relu(self.fc1(task_ids))
        x = F.leaky_relu(self.fc2(x))
        matrix = self.fc3(x)
        bias = self.fc4(x)
        task_ids_enc_resized = matrix.view(-1, self.latent_size, self.latent_size)
        task_ids_enc_resized = torch.softmax(task_ids_enc_resized, 1)
        return task_ids_enc_resized, bias

# class Translator(nn.Module):
#     def __init__(self, n_dim_coding, p_coding, device):
#         super().__init__()
#         self.n_dim_coding = n_dim_coding
#         self.p_coding = p_coding
#         self.device = device
#         # self.latent_size = latent_size
#
#         self.fc1 = nn.Linear(n_dim_coding, n_dim_coding * 2)
#         self.fc2 = nn.Linear(n_dim_coding * 2, n_dim_coding * 3)
#         self.fc3 = nn.Linear(n_dim_coding * 3, n_dim_coding)
#
#     def forward(self, task_id):
#         codes = (task_id * self.p_coding) % (2 ** self.n_dim_coding)
#         x = unpackbits(codes, self.n_dim_coding).to(self.device)
#         # x = torch.cat([x, task_ids], dim=1)
#         x = torch.sigmoid(self.fc1(x))
#         x = torch.sigmoid(self.fc2(x))
#         x = torch.sigmoid(self.fc3(x))
#         return x

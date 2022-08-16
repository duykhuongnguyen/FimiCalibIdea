# import config as CFG
import numpy as np
import torch
import torch.nn as nn
from models.modules import *


class GRUMTL(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_class, device=None, mean=None, std=None):
        super(GRUMTL, self).__init__()

        self.data_mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self.data_std = torch.tensor(std, dtype=torch.float32, device=device)  
        self.pri_enc = nn.Sequential()
        for i in range(n_class):
            self.pri_enc.add_module(f"Pri Enc {i}", SeriesEncoder(input_dim, hidden_dim))

        self.sha_enc = SeriesEncoder(input_dim * n_class, hidden_dim)
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim * 2, num_layers=2, batch_first=True, bidirectional=True)
        self.dec = nn.Sequential()
        for i in range(n_class):
            self.dec.add_module(f"DEC {i}", nn.Linear(hidden_dim * 4, output_dim))

    def forward(self, input, lab=None, noise=None):
        # shape of images: (batch, channel, height, width)
        _, M, _, _ = input.shape

        input_cat, latent = [], []
        for i in range(M):
            input_i = input[:, i, :, :]
            input_i = (input_i - self.data_mean[i]) / self.data_std[i]
            input_cat.append(input_i)
            latent_i = self.pri_enc[i](input_i).transpose(0, 1).contiguous()
            latent.append(latent_i)
        latent = torch.stack(latent, dim=1)
        input_cat = torch.cat(input_cat, dim=2)

        latent_sha = self.sha_enc(input_cat).transpose(0, 1).contiguous()
        
        output = []
        for i in range(M):
            latent_i = latent[:, i, :, :]
            latent_i, _ = self.gru(latent_i)

            calib_output_i = self.dec[i](latent_i)
            calib_output_i = calib_output_i * self.data_std[i] + self.data_mean[i]
            output.append(calib_output_i)
        output = torch.stack(output, dim=1)

        return output, None


if __name__ == '__main__':
    import numpy as np
    mean = np.random.rand(5,4)
    std = np.random.rand(5,4)
    model = MSJF(4, 64, 4, 5, "cpu", mean, std)
    print(model)

    input = torch.randn(2, 5, 7, 4)
    calib_outs = model(input)

    # print(calib_outs.shape)

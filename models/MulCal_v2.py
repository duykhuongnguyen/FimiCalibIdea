import config as CFG
import torch.nn as nn
from models.modules import *

class MulCal(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_class, device, mean, std, noise=False, new_gen=False):
        super(MulCal, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.data_mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self.data_std = torch.tensor(std, dtype=torch.float32, device=device)
        print(self.data_mean)
        self.n_class = n_class
        self.noise = noise
        self.new_gen = new_gen

        self.extractor = SeriesEncoder(input_dim, hidden_dim - 2) if noise else SeriesEncoder(input_dim, hidden_dim)
        self.identity = IdentityLayer_v2(hidden_dim * 2, hidden_dim, n_class)
        self.seperate_module = IdentityMergingModule(self.n_class, self.hidden_dim, self.hidden_dim * 2, 3)
        self.calib = IdentityAwaredCalibModule_v2(device, hidden_dim * 2, output_dim)
        self.discriminator = Discriminator(device, input_dim, hidden_dim, output_dim)
        self.cond_to_latent = nn.LSTM(input_size=input_dim,
                                      hidden_size=hidden_dim,
                                      bidirectional=False,
                                      batch_first=True)
        self.latent_to_cal = nn.LSTMCell(input_size=hidden_dim + 4,
                                         hidden_size=hidden_dim)
        self.ctx_2_cal = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim * 2, out_features=output_dim)
        )

    def forward(self, input, label, noise=None):
        '''
        input with shape (N, M, L, H), in which:
            N - batch size
            M - number of device
            L - sequence length
            H - input features
ition.size(0), CFG.noise_dim)),
                                            devi        '''

        _, M, _, _ = input.shape

        latent_inputs = []
        identity_latents = []
        for i in range(M):
            input_i = input[:, i, :, :]
            label_i = label[:, i, :]
            input_i = (input_i - self.data_mean[i]) / self.data_std[i]
            if self.new_gen:
                condition_latent, (hl, _) = self.cond_to_latent(input_i)
            else:
                latent_input_i = self.extractor(input_i).transpose(0, 1).contiguous()
            if self.noise:
                if self.new_gen:
                    lc_input = torch.cat((hl.squeeze(0), noise), dim=1)
                    condition_tilde = []                                                       
                    for _ in range(self.hidden_dim):
                        hl_tilde, _ = self.latent_to_cal(lc_input)          
                        lc_input = torch.cat((hl_tilde, noise), dim=1)
                        hl_coeff = torch.bmm(condition_latent, hl_tilde.unsqueeze(2)).squeeze(2)                                
                        hl_coeff = torch.softmax(hl_coeff, dim=1)       
                        hl_context = torch.bmm(hl_coeff.unsqueeze(1), condition_latent).squeeze(1)
                        hl_fin = torch.cat((hl_tilde, hl_context), dim=1)
                        condition_tilde.append(hl_fin)
                    latent_input_i = torch.stack(condition_tilde, dim=1)
                else:
                    latent_input_i = torch.cat((latent_input_i, noise), dim=2)
            identity_latent_input_i = self.identity(label_i)
            latent_inputs.append(latent_input_i)
            identity_latents.append(identity_latent_input_i)
        
        latent_input = torch.stack(latent_inputs, dim=1)
        # print(latent_input.shape)
        identity_latent = torch.stack(identity_latents, dim=1)
        # print(identity_latent.shape)

        merged_inputs, sep_indicator = self.seperate_module(latent_input, identity_latent)
        # print(merged_inputs.shape)

        calib_outs = []
        for i in range(M):
            # Calibration
            input_i = input[:, i, :, :]
            input_i = (input_i - self.data_mean[i]) / self.data_std[i]
            # latent_input_i = latent_input_i.permute(1, 0, 2).contiguous()
            merged_input_i = merged_inputs[:, i, :, :].transpose(0, 1).contiguous()
            identity_latent_input_i = identity_latent[:, i, :]
            calib_output = self.calib(merged_input_i, identity_latent_input_i)
            calib_output = calib_output * self.data_std[i] + self.data_mean[i]
            calib_outs.append(calib_output)
            # print(self.discriminator(input_i, calib_output))

        calib_outs = torch.stack(calib_outs, dim=1)

        return calib_outs, sep_indicator


if __name__ == '__main__':
    import numpy as np
    mean = np.random.rand(5,4)
    std = np.random.rand(5,4)
    model = MulCal(CFG.input_dim, CFG.hidden_dim, CFG.output_dim, CFG.n_class, 'cpu', mean, std)
    print(model)

    input = torch.randn(2, 5, 7, CFG.input_dim)
    label = torch.randn(2, 5, 5)
    calib_outs = model(input, label)

    print(calib_outs.shape)

import numpy as np
import torch

import config as CFG
from models.MulCal_v2 import MulCal


rs = np.random.RandomState(1368)


mean = np.random.rand(5,4)
std = np.random.rand(5,4)
model = MulCal(CFG.input_dim, CFG.hidden_dim, CFG.output_dim, CFG.n_class, 'cpu', mean, std, noise=True)
print(model)

input = torch.randn(2, 5, 7, CFG.input_dim)
label = torch.randn(2, 5, 5)               
noise_batch = torch.tensor(rs.normal(0, 1, (2, 7, 4)), dtype=torch.float32)

calib_outs = model(input, label, noise_batch)           
print(calib_outs[0].shape)

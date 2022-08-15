import argparse
import numpy as np
import torch

import config as CFG
import utils
from Data.calib_loader import CalibDataset
from models.MulCal_v2 import MulCal
from models.msjf import MSJF


# Fixing random seeds
torch.manual_seed(1368)
rs = np.random.RandomState(1368)


# Prepare dataset
X_train, y_train, lab_train, X_val, y_val, lab_val, X_test, y_test, lab_test = utils.prepare_multicalib_dataset(single=False)
X_mean = X_train.mean(axis=0).mean(axis=1)
X_std = X_train.std(axis=0).mean(axis=1)

test_loader = torch.utils.data.DataLoader(CalibDataset(X_test, y_test, lab_test), batch_size=CFG.batch_size, shuffle=False)

# Loader
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = MulCal(CFG.input_dim, CFG.hidden_dim, CFG.output_dim, CFG.n_class, device, X_mean, X_std, noise=True, new_gen=False)
model.load_state_dict(torch.load(f"./logs/checkpoints/{self.args.name}_best.pt"))
model.eval()

# Inference
def infer(x, lab):
    noise_batch = torch.tensor(torch.tensor(rs.normal(0, 1, (x.size(0), CFG.input_timestep, CFG.noise_dim)), device=device, dtype=torch.float32)
    pred, _ = model(x, lab, noise_batch)

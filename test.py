import argparse
import numpy as np
import matplotlib.pyplot as plt
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
model.load_state_dict(torch.load(f"./logs/checkpoints/multi_best.pt"))
model.to(device)
model.eval()

# Inference function
def infer(x, lab):
    noise_batch = torch.tensor(rs.normal(0, 1, (x.size(0), CFG.input_timestep, CFG.noise_dim)), device=device, dtype=torch.float32)
    pred, _ = model(x, lab, noise_batch)
    return pred

# Inference
x, y, lab = next(iter(test_loader))
x = x.to(device)
lab = lab.to(device)
preds = infer(x, lab)

x = x.detach().cpu()
lab = lab.detach().cpu()
preds = preds.detach().cpu()

# Plot
atts = ['PM2_5', 'PM10', 'humidity']
ids = ['1', '14', '20', '27', '30']
fig, ax = plt.subplots(len(atts), len(ids), figsize=(20, 25))

for i, idx in enumerate(ids):                                  
    for j, att in enumerate(atts):                             
        x_i = x[:, i, 0, j]                          
        y_i = y[:, i, 0, j]                          
        pred_i = preds[:, i, 0, j]                             

        rn_test = range(x_i.shape[0])                          
        ax[j, i].plot(rn_test, x_i, 'g', label='raw')          
        ax[j, i].plot(rn_test, y_i, 'b', label='gtruth')       
        ax[j, i].plot(rn_test, pred_i, 'r', label='calibrated')
        ax[j, i].legend(loc='best')                            
        ax[j, i].set_title(f"device: {idx}")                   
        ax[j, i].set_xlabel("time")                            
        ax[j, i].set_ylabel(att)                               
    
fig.savefig(f"./logs/test.png")
                                                                                                   
# for x, y, lab in test_loader:
#     x = x.to(device)
#     y = y.to(device)
#     lab = lab.to(device)

#     calib = infer(x, lab)
#     print(calib.shape)

import argparse
from single_calib import SingleCalibModel
import utils
# from forgan import ForGAN
# from multicalib import MultiCalibModel
import config as CFG

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # mg for Mackey Glass and itd = Internet traffic dataset (A5M)
    ap.add_argument("--metric", metavar='', dest="metric", type=str, default='mse',
                    help="metric to save best model - mae or rmse or kld")
    ap.add_argument("--es", metavar='', dest="early_stop", type=int, default=50,
                    help="early stopping patience")
    ap.add_argument("--ssa", metavar='', dest="ssa", type=bool, default=False,
                    help="use ssa preprocessing")
    ap.add_argument("--type", metavar='', dest="train_type", type=str, default='train',
                    help="train")
    ap.add_argument("--name", metavar='', dest="name", type=str, default='multi',
                    help="name of the model - project")
    ap.add_argument("--usen", metavar='', dest="use_n", type=int, default=0,
                    help="whether to use n model (0) or only one (1) or propose model (2)")
    opt = ap.parse_args()

    # x_train, y_train, lab_train, x_val, y_val, lab_val, x_test, y_test, lab_test = utils.prepare_multicalib_dataset()
    x_train, y_train, lab_train, x_val, y_val, lab_val, x_test, y_test, lab_test = utils.prepare_multicalib_dataset(single=False)
   
    x_mean = x_train.mean(axis=0)
    # x_mean = x_mean.mean(axis=0)
    x_mean = x_mean.mean(axis=1)
    print(x_mean.shape)
    # print(x_mean.shape)
    x_std = x_train.std(axis=0)
    # x_std = x_std.mean(axis=0)
    x_std = x_std.mean(axis=1)
    print(x_std.shape)
    opt.data_mean = x_mean
    opt.data_std = x_std

    seed_everything(911)
    opt.device_ids = CFG.devices
    opt.attributes = CFG.attributes

    if opt.use_n == 0:
        from multicalib_Nmodel import MultiCalibModel
        model = MultiCalibModel(opt, x_train, y_train, lab_train, x_val, y_val, lab_val, x_test, y_test, lab_test, use_n=True)
    elif opt.use_n == 1:
        from multicalib_Nmodel import MultiCalibModel
        model = MultiCalibModel(opt, x_train, y_train, lab_train, x_val, y_val, lab_val, x_test, y_test, lab_test, use_n=False)
    elif opt.use_n == 10:
        from multicalib import MultiCalibModel
        model = MultiCalibModel(opt, x_train, y_train, lab_train, x_val, y_val, lab_val, x_test, y_test, lab_test, baseline=1)
    elif opt.use_n == 11:
        from multicalib import MultiCalibModel
        model = MultiCalibModel(opt, x_train, y_train, lab_train, x_val, y_val, lab_val, x_test, y_test, lab_test, baseline=2)
    elif opt.use_n == 2:
        from multicalib import MultiCalibModel
        model = MultiCalibModel(opt, x_train, y_train, lab_train, x_val, y_val, lab_val, x_test, y_test, lab_test)
    else:
        from multicalib import MultiCalibModel
        model = MultiCalibModel(opt, x_train, y_train, lab_train, x_val, y_val,
 lab_val, x_test, y_test, lab_test, gan_loss=True)
        model.train_gan()
    
    if opt.train_type == 'train':
        model.train()
        model.test()
    else:
        model.test()


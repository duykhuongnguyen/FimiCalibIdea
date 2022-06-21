import numpy as np
import torch
import torch.nn as nn
from utils import EarlyStopping, MetricLogger
from util.losses import ConstrastiveLoss
from models.MulCal_v2 import MulCal
from models.modules import *
from Data.calib_loader import CalibDataset
import config as CFG
import matplotlib.pyplot as plt


class MultiCalibModel:
    def __init__(self, args, x_train, y_train, lab_train, x_val, y_val, lab_val, x_test, y_test, lab_test, devices=CFG.devices, use_n=False, gan_loss=False):
        self.args = args
        self.train_loader = torch.utils.data.DataLoader(CalibDataset(x_train, y_train, lab_train), batch_size=CFG.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(CalibDataset(x_val, y_val, lab_val), batch_size=CFG.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(CalibDataset(x_test, y_test, lab_test), batch_size=CFG.batch_size, shuffle=False)
        
        self.x_test = x_test
        self.y_test = y_test

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.es = EarlyStopping(self.args.early_stop)
        self.gan_loss = gan_loss

        print(f"Use device: {self.device}")
        print("*****  Hyper-parameters  *****")
        for k, v in vars(args).items():
            print("{}:\t{}".format(k, v))
        print("************************")

        self.model = MulCal(CFG.input_dim, CFG.hidden_dim, CFG.output_dim, CFG.n_class, self.device, self.args.data_mean, self.args.data_std, noise=gan_loss)
        self.model.to(self.device)
        print("\nNetwork Architecture\n")
        print(self.model)
        print("\n************************\n")

        if self.gan_loss:
            self.discriminator = Discriminator(self.device, CFG.input_dim, CFG.hidden_dim, CFG.output_dim)
    
    def train(self):
        best_mse = np.inf

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=CFG.lr)
        criteria = nn.MSELoss()
        criteria = criteria.to(self.device)
        criteria_cons = ConstrastiveLoss()  

        log_dict = {}
        logger = MetricLogger(self.args, tags=['train', 'val'])
        
        for epoch in range(CFG.epochs):
            self.model.train()
            mse_train, mae_train, mape_train = 0, 0, 0
            cnt = 0
            for x, y, lab in self.train_loader:
                cnt += 1
                x = x.to(self.device)
                y = y.to(self.device)
                lab = lab.to(self.device)
                self.model.zero_grad()

                pred, sep_indicator = self.model(x, lab)
                # print(pred.shape)
                # print(y.shape)
                loss = criteria(pred, y)
                loss_cons = criteria_cons(sep_indicator)
                loss_cons.backward(retain_graph=True)

                mae = torch.mean(torch.abs(pred - y))
                # mape = torch.mean(torch.abs((pred - y) / y)) * 100
                # print(mape)

                mse_train += loss
                mae_train += mae
                # mape_train += mape
                loss.backward()
                optimizer.step()
            
            mse_train /= cnt
            mae_train /= cnt
            # mape_train /= cnt

            log_dict['train/mse'] = mse_train
            log_dict['train/mae'] = mae_train
            # log_dict['train/mape'] = mape_train
            # validation
            self.model.eval()
            mse, mae, mape = 0, 0, 0
            cnt = 0
            for x, y, lab in self.val_loader:
                cnt += 1
                x = x.to(self.device)
                y = y.to(self.device)
                lab = lab.to(self.device)
                pred, _ = self.model(x, lab)

                mse += criteria(pred, y)
                mae += torch.abs(pred - y).mean()
                # mape += torch.mean(torch.abs((pred - y) / y)) * 100
            mse /= cnt
            mae /= cnt
            # mape /= cnt

            log_dict['val/mse'] = mse
            log_dict['val/mae'] = mae
            # log_dict['val/mape'] = mape
            logger.log_metrics(epoch, log_dict)
            # print(log_dict)
            print(f"Epoch: {epoch+1:3d}/{CFG.epochs:3d}, MSE_val: {mse:.4f}, MAE_val: {mae:.4f}")
            self.es(mse)

            if mse < best_mse:
                best_mse = mse
                torch.save(self.model.state_dict(), f"./logs/checkpoints/{self.args.name}_best.pt")
            else: 
                torch.save(self.model.state_dict(), f"./logs/checkpoints/{self.args.name}_last.pt")
                # self.model.load_state_dict(torch.load(f"./logs/checkpoints/{self.args.name}_best.pt"))
                if (self.es.early_stop):
                    print("Early stopping")
                    break
    
    def train_gan(self):
        best_mse = np.inf

        optimizer_g = torch.optim.RMSprop(self.model.parameters(), lr=self.opt.lr)
        optimizer_d = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.opt.lr)
        adversarial_loss = nn.BCELoss()
        generator_loss = nn.MSELoss()

        adversarial_loss = adversarial_loss.to(self.device)
        generator_loss = generator_loss.to(self.device)  

        log_dict = {}
        logger = MetricLogger(self.args, tags=['train', 'val'])
        
        for epoch in range(CFG.epochs):
            # self.model.train()
            d_loss = 0
            for _ in range(self.args.d_iter):
                # train discriminator on real data
                for x, y, lab in self.train_loader:
                    condition = x.to(self.device)
                    real_data = y.to(self.device)
                    lab = lab.to(self.device)
                
                    self.discriminator.zero_grad()
                    d_real_decision = self.discriminator(real_data, condition)
                    d_real_loss = 1/ 2 * adversarial_loss(d_real_decision,
                                                torch.full_like(d_real_decision, 1, device=self.device))
                    d_real_loss.backward()
                    optimizer_d.step()

                    d_loss += d_real_loss.detach().cpu().numpy()
                    # train discriminator on fake data
                    noise_batch = torch.tensor(rs.normal(0, 1, (condition.size(0), CFG.noise_dim)),
                                            device=self.device, dtype=torch.float32)
                    x_fake = self.generator(noise_batch, condition).detach()
                    d_fake_decision = self.discriminator(x_fake, condition)
                    d_fake_loss = 1/2 * adversarial_loss(d_fake_decision,
                                                torch.full_like(d_fake_decision, 0, device=self.device))
                    d_fake_loss.backward()
                    optimizer_d.step()
                    
                    d_loss += d_fake_loss.detach().cpu().numpy()

            d_loss = d_loss / (2 * self.opt.d_iter)

            self.generator.zero_grad()
            noise_batch = torch.tensor(rs.normal(0, 1, (self.opt.batch_size, CFG.noise_dim)), device=self.device,
                                       dtype=torch.float32)

            x_fake = self.generator(noise_batch, condition)
            # print(x_fake)
            d_g_decision = self.discriminator(x_fake, condition)
            
            # Mackey-Glass works best with Minmax loss in our expriements while other dataset
            # produce their best result with non-saturated loss
            # if opt.dataset == "mg" or opt.dataset == 'aqm':
            g_loss = 1/2 * adversarial_loss(d_g_decision, torch.full_like(d_g_decision, 1, device=self.device))
            # else:
            #     g_loss = -1 * adversarial_loss(d_g_decision, torch.full_like(d_g_decision, 0, device=self.device))
            g_loss.backward()
            optimizer_g.step()

            g_loss = g_loss.detach().cpu().numpy()

            # Validation
            noise_batch = torch.tensor(rs.normal(0, 1, (x_val.size(0), CFG.noise_dim)), device=self.device,
                                       dtype=torch.float32)
            preds = self.generator(noise_batch, x_val).detach().cpu().numpy()

            kld = utils.calc_kld(preds, y_val, self.opt.hist_bins, self.opt.hist_min, self.opt.hist_max)
            rmse =  np.sqrt(np.square(preds - y_val).mean())
            mae = np.abs(preds - y_val).mean()

            if self.opt.metric == 'kld': 
                self.es(kld)
                if self.es.early_stop:
                    break
                if kld <= best_kld and kld != np.inf:
                    best_kld = kld
                    print("step : {} , KLD : {}, RMSE : {}, MAE: {}".format(step, best_kld,
                                                                rmse, mae))
                    torch.save({
                        'g_state_dict': self.generator.state_dict()
                    }, "./{}/{}_best.torch".format(self.opt.dataset, self.opt.name))
            elif self.opt.metric == 'rmse':
                self.es(rmse)
                if self.es.early_stop:
                    break
                if rmse <= best_rmse and rmse != np.inf:
                    best_rmse = rmse
                    print("step : {} , KLD : {}, RMSE : {}, MAE: {}".format(step, kld,
                                                                rmse, mae))
                    torch.save({
                        'g_state_dict': self.generator.state_dict()
                    }, "./{}/{}_best.torch".format(self.opt.dataset, self.opt.name))
            else:
                self.es(mae)
                if self.es.early_stop:
                    break
                if mae <= best_mae and mae != np.inf:
                    best_mae = mae
                    print("step : {} , KLD : {}, RMSE : {}, MAE: {}".format(step, kld,
                                                                rmse, mae))
                    torch.save({
                        'g_state_dict': self.generator.state_dict()
                    }, "./{}/{}_best.torch".format(self.opt.dataset, self.opt.name))

            if step % 100 == 0:
                print(YELLOW_TEXT + BOLD + "step : {} , d_loss : {} , g_loss : {}".format(step, d_loss, g_loss) + ENDC)
                torch.save({
                    'g_state_dict': self.generator.state_dict(), 
                    'd_state_dict': self.discriminator.state_dict(), 
                }, "./{}/{}_checkpoint.torch".format(self.opt.dataset, self.opt.name))
    
    def test(self):
        self.model.load_state_dict(torch.load(f"./logs/checkpoints/{self.args.name}_best.pt"))
        
        self.model.eval()
        mse, mae, mape = 0, 0, 0
        cnt = 0
        preds = []
        gtruths = []
        for x, y, lab in self.test_loader:
            cnt += 1
            x = x.to(self.device)
            y = y.to(self.device)
            lab = lab.to(self.device)
            pred, _ = self.model(x, lab)

            preds.append(pred.cpu().detach().numpy())
            gtruths.append(y.cpu().detach().numpy())
            # mse += torch.mean((pred - y) ** 2)
            # mae += torch.abs(pred - y).mean()
            # mape += torch.abs(pred - y).mean() / y.mean() * 100

        preds = np.concatenate(preds, axis=0)
        gtruths = np.concatenate(gtruths, axis=0)
        N, M, L, H = preds.shape
        preds_flt = preds.transpose(1,0,2,3).reshape(M, -1)
        gtruths_flt =  gtruths.transpose(1,0,2,3).reshape(M, -1)

        print(preds.shape)
        print(gtruths.shape)
        mse = np.square(preds_flt - gtruths_flt).mean(axis=1)
        mae = np.abs(preds_flt - gtruths_flt).mean(axis=1)
        mape = np.abs(preds_flt - gtruths_flt).mean(axis=1) / gtruths_flt.mean(axis=1) * 100
        # mse /= cnt
        # mae /= cnt
        # mape /= cnt

        print(f"MSE_test: {mse}, MSE mean: {mse.mean()} \nMAE_test: {mae}, MAE mean: {mae.mean()}, \nMAPE_test: {mape}, MAPE mean: {mape.mean()}")
    
        ids = self.args.device_ids[1:]
        print(ids)
        atts = self.args.attributes
        fig, ax = plt.subplots(len(atts), len(ids), figsize=(20, 25))
        for i, idx in enumerate(ids):
            for j, att in enumerate(atts):
                x_i = self.x_test[:, i, 0, j]
                y_i = self.y_test[:, i, 0, j]
                pred_i = preds[:, i, 0, j]

                rn_test = range(x_i.shape[0])
                ax[j, i].plot(rn_test, x_i, 'g', label='raw')
                ax[j, i].plot(rn_test, y_i, 'b', label='gtruth')
                ax[j, i].plot(rn_test, pred_i, 'r', label='calibrated')
                ax[j, i].legend(loc='best')
                ax[j, i].set_title(f"device: {idx}")
                ax[j, i].set_xlabel("time")
                ax[j, i].set_ylabel(att)

        fig.savefig(f"./logs/figures/{self.args.name}_test.png")

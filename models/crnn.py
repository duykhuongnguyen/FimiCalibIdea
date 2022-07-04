# import config as CFG
import numpy as np
import torch
import torch.nn as nn
# from models.modules import *


class CRNN(nn.Module):

    def __init__(self, input_dim, output_dim, n_class, device=None, mean=None, std=None,
                 map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False):
        super(CRNN, self).__init__()

        self.data_mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self.data_std = torch.tensor(std, dtype=torch.float32, device=device)  
        self.cnn = nn.Sequential()
        for i in range(n_class):
            self.cnn.add_module(f"CNN {i}", self._cnn_backbone(1, 7, input_dim, leaky_relu, i))

        self.decnn = nn.Sequential()
        for i in range(n_class):
            self.decnn.add_module(f"DECNN {i}", self._decnn_backbone(1, 7, input_dim, leaky_relu, i))

        self.map_to_seq = nn.Linear(128 * 7, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, 7)

    def _cnn_backbone(self, img_channel, img_height, img_width, leaky_relu, dev):
        channels = [1, 64, 128, 128, 128]
        kernel_sizes = [3, 3, 3, 3]
        strides = [1, 1, 1, 1]
        paddings = [1, 1, 1, 1]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = channels[i]
            output_channel = channels[i+1]

            cnn.add_module(
                f'conv{i}_{dev}',
                nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}_{dev}', nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}_{dev}', relu)

        # size of image: (channel, height, width) = (img_channel, img_height, img_width)
        conv_relu(0)
        cnn.add_module(f'pooling0_{dev}', nn.MaxPool2d(kernel_size=2, stride=1, padding=1, dilation=2))
        # (64, img_height // 2, img_width // 2)

        conv_relu(1)
        cnn.add_module(f'pooling1_{dev}', nn.MaxPool2d(kernel_size=2, stride=1, padding=1, dilation=2))
        # (128, img_height // 4, img_width // 4)

        conv_relu(2)
        cnn.add_module(f'pooling2_{dev}', nn.MaxPool2d(kernel_size=2, stride=1, padding=1, dilation=2))  # (256, img_height // 8, img_width // 4)

        conv_relu(3)
        # conv_relu(4, batch_norm=True)
        # conv_relu(5, batch_norm=True)
        # cnn.add_module('pooling3', nn.MaxPool2d(kernel_size=2, stride=1, padding=1, dilation=2))  # (512, img_height // 16, img_width // 4)

        # conv_relu(6)  # (512, img_height // 16 - 1, img_width // 4 - 1)

        # output_channel, output_height, output_width = \
        #     channels[-1], img_height // 16 - 1, img_width // 4 - 1
        
        return cnn

    def _decnn_backbone(self, img_channel, img_height, img_width, leaky_relu, dev):                                                              
        channels = [128, 128, 128, 64, 1] # [1, 64, 128, 128, 128]                                                                          
        kernel_sizes = [3, 3, 3, 3]                                                                                                       
        strides = [1, 1, 1, 1]                                                                                                            
        paddings = [1, 1, 1, 1]                                                                                                           
                                                                                                                                      
        cnn = nn.Sequential()                                                                                                             
                                                                                                                                      
        def conv_relu(i, batch_norm=False):                                                                                               
            # shape of input: (batch, input_channel, height, width)                                                                       
            input_channel = channels[i]                                                                                                   
            output_channel = channels[i+1]                                                                                                
                                                                                                                                      
            cnn.add_module(                                                                                                               
                f'deconv{i}_{dev}',                                                                                                               
                nn.ConvTranspose2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])                                        
            )                                                                                                                             
                                                                                                                                      
            if batch_norm:                                                                                                                
                cnn.add_module(f'debatchnorm{i}_{dev}', nn.BatchNorm2d(output_channel))                                                           
                                                                                                                                      
            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)                                               
            cnn.add_module(f'derelu{i}_{dev}', relu)                                                                                              
                                                                                                                                      
        # size of image: (channel, height, width) = (img_channel, img_height, img_width)                                                  
        conv_relu(0)                                                                                                                      
        cnn.add_module(f'depooling0_{dev}', nn.MaxPool2d(kernel_size=2, stride=1, padding=1, dilation=2))                                          
        # (64, img_height // 2, img_width // 2)                                                                                           
                                                                                                                                      
        conv_relu(1)                                                                                                                      
        cnn.add_module(f'depooling1_{dev}', nn.MaxPool2d(kernel_size=2, stride=1, padding=1, dilation=2))                                          
        # (128, img_height // 4, img_width // 4)                                                                                          
                                                                                                                                      
        conv_relu(2)                                                                                                                      
        cnn.add_module(f'depooling2_{dev}', nn.MaxPool2d(kernel_size=2, stride=1, padding=1, dilation=2))  # (256, img_height // 8, img_width // 4)
                                                                                                                                      
        conv_relu(3)        

        return cnn

    def forward(self, input, lab=None):
        # shape of images: (batch, channel, height, width)
        _, M, _, _ = input.shape

        latent = []
        for i in range(M):
            input_i = input[:, i, :, :].unsqueeze(1)
            input_i = (input_i - self.data_mean[i]) / self.data_std[i]
            conv_i = self.cnn[i](input_i)
            latent.append(conv_i)
        latent = torch.stack(latent, dim=1)
        
        output, output_deconv = [], []
        for i in range(M):
            conv_i = latent[:, i, :, :]
            deconv_i = self.decnn[i](conv_i).squeeze(1)
            deconv_i = deconv_i * self.data_std[i] + self.data_mean[i]
            output_deconv.append(deconv_i)

            batch, channel, height, width = conv_i.size()
            conv_i = conv_i.view(batch, channel * height, width)
            conv_i = conv_i.permute(2, 0, 1)  # (width, batch, feature)
            seq_i = self.map_to_seq(conv_i)

            recurrent, _ = self.rnn1(seq_i)
            recurrent, _ = self.rnn2(recurrent)
            calib_output_i = self.dense(recurrent).permute(1, 2, 0)
            calib_output_i = calib_output_i * self.data_std[i] + self.data_mean[i]
            output.append(calib_output_i)
        output = torch.stack(output, dim=1)
        output_deconv = torch.stack(output_deconv, dim=1)

        return output, output_deconv


if __name__ == '__main__':
    import numpy as np
    mean = np.random.rand(5,4)
    std = np.random.rand(5,4)
    model = CRNN(4, 4, 5, "cpu")
    print(model)

    input = torch.randn(2, 5, 7, 4)
    calib_outs = model(input)

    # print(calib_outs.shape)

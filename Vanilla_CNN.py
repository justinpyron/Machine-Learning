import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


class Conv_Block(nn.Module):
    '''
    Block that performs the following operation:
    input --> dropout --> conv_3D --> conv_3D
            --> ReLU --> BatchNorm --> MaxPool
    Note: It appears that, contrary to the original BN
          paper, applying BN *after* the
          nonlinearity gives better results
          
    Inputs:
    <in_channels>: integer giving the number of channels of input
    <out_channels>: integer giving the number of channels of output
    <dropout_prob>: the probability of zero-ing inputs
    <do_maxpool>: True/False indicating if a maxpool should be done 
                    at end of layer
    '''
    def __init__(self, in_channels, out_channels, dropout_prob, do_maxpool):
        super(Conv_Block, self).__init__()
        self.dropout_prob = dropout_prob
        self.do_maxpool = do_maxpool
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = F.dropout(x, p=self.dropout_prob)
        x = self.conv_2(self.conv_1(x))
        x = F.relu(x)
        x = self.bn(x)
        if self.do_maxpool == True:
            x = F.max_pool2d(x, kernel_size=2)
        return x


class CNN(nn.Module):
    '''
    Inputs:
    <input_shape>: the dimensions of input image of form (C, d_1, d_2):
    			   C = number of channels
    			   D_1, D_2 = spatial dimensions
    <filter_list>: a list containing how many filters (i.e. channels) to
                   use in each Conv_Block
    <maxpool_list>: a list indicating whethere a maxpool should be performed
                    in each Conv_Block. 0 = no maxpool, 1 = maxpool
    <fc_list>: a list indicating how many neurons to use in each fully connected layer
    <dropout_prob>: the probability of zero-ing neurons
    Note: The length of <filter_list> and <maxpool_list> indicate
          how many Conv_Blocks are in the model.
          The length of <fc_list> indicates how many FC layers are in
          the model.
    '''
    def __init__(self,
                 input_shape,
                 filter_list,
                 maxpool_list,
                 fc_list,
                 Conv_to_FC_method,
                 dropout_prob):
        super(CNN, self).__init__()
        self.dropout_prob = dropout_prob
        self.num_conv_blocks = len(filter_list)
        self.num_fc_layers = len(fc_list)
        C, D_1, D_2 = input_shape
        
        # Convolutional modules
        # ===============================================
        self.conv_modules = nn.ModuleList()
        for i in range(self.num_conv_blocks):
            in_channels = C if i==0 else filter_list[i-1]
            out_channels = filter_list[i]
            drop_p = 0 if i==0 else dropout_prob # Don't apply dropout to input
            do_max = maxpool_list[i]
            self.conv_modules.append(Conv_Block(in_channels, 
            									out_channels, 
            									drop_p, 
            									do_max))
        
        # Fully-Connected Modules
        # ===============================================
        num_maxpools = sum(maxpool_list)
        reduction_factor = filter_list[-1]/( (2**num_maxpools)**C )
        # above calculation:
        # [ (2^num_maxpools)^num_spatial_dim ] * num_filters_in_last_conv_layer

        start_neurons = np.prod(input_shape)*reduction_factor
        
        self.fc_modules = nn.ModuleList()
        for i in range(self.num_fc_layers):
            in_neurons = start_neurons if i==0 else fc_list[i-1]
            out_neurons = fc_list[i]
            self.fc_modules.append(nn.Linear(int(in_neurons), int(out_neurons)))


    def forward(self, x):
        # Convolutional layers
        # ===============================================
        for i in range(self.num_conv_blocks):
            x = self.conv_modules[i](x)

        # Transition from Conv to FC layers
        # ===============================================
        x = x.view(x.size(0),-1) # flatten
        
        # Fully-connected layers
        # ===============================================
        for i in range(self.num_fc_layers):
            x = F.dropout(x, p=self.dropout_prob)
            x = self.fc_modules[i](x)
            if i < self.num_fc_layers-1:
                # Don't use non-linearity on last FC layer
                x = F.relu(x)
        x = F.log_softmax(x, dim=1)
        
        return x


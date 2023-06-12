import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import surrogate
import matplotlib.pyplot as plt
import random
import imageio
import snntorch.spikeplot as splt
from pathlib import Path
from data_loader.testing import TestDatabase
from tqdm import tqdm, trange
import torch
from pathlib import Path
from data_loader.testing import TestDatabase
import tonic
import os
from model import getNetwork
#from utils import moveToGPUDevice
from utils.gpu import moveToGPUDevice
from config.utils import getTestConfigs


logdir = os.path.join(os.getcwd(), 'logs/test')
config = os.path.join(os.getcwd(), 'test_config.yaml')
write = 'store_true'

configs = getTestConfigs(logdir, config)

device = configs['general']['hardware']['gpuDevice']
dtype = configs['general']['model']['dtype']

net = getNetwork(configs['general']['model']['type'],
                configs['general']['simulation'])

general_config = configs['general']
log_config = configs['log']


def loadNetFromCheckpoint(net, general_config, log_config):
    ckpt_file = general_config['model']['CkptFile']
    print('Loading checkpoint from {}'.format(ckpt_file))
    assert ckpt_file
    checkpoint = torch.load(ckpt_file,
            map_location=general_config['hardware']['gpuDevice'])

    net = getNetwork(general_config['model']['type'],
            general_config['simulation'])
    net.load_state_dict(checkpoint['model_state_dict'])
    moveToGPUDevice(net, device, dtype)
    log_config.copyModelFile(net)

    return net

net = loadNetFromCheckpoint(net, general_config, log_config)

for name, parameters in net.named_parameters():
    print(name, ':', parameters.size())

data = torch.randn((8, 20, 2, 180, 240)).to('cuda')

data = data.permute(0, 2, 3, 4, 1)
out = net(data)
print(out.size())

def get_parameter_number(net): 
    total_num = sum(p.numel() for p in net.parameters()) 
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad) 
    return {'Total': total_num, 'Trainable': trainable_num}

print(get_parameter_number(net))

# conv1 = torch.nn.Conv2d(2, 16, 3, stride=2)
# conv2 = torch.nn.Conv2d(16, 32, 3, stride=2)
# conv3 = torch.nn.Conv2d(32, 64, 3, stride=2)
# conv4 = torch.nn.Conv2d(64, 128, 3, stride=2)
# conv5 = torch.nn.Conv2d(128, 256, 3, stride=1)

# data = torch.randn((8, 2, 180, 240))
# out = conv5(conv4(conv3(conv2(conv1(data)))))
# print(out.size())

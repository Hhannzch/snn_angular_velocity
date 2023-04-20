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

data_dir = '/home/chuhan/chuhan/rotation_work/snn_angular_velocity/data'

train_database = TestDatabase(data_dir)
train_loader = torch.utils.data.DataLoader(
                train_database,
                batch_size=8,
                shuffle=False)

random.seed(1120)
np.random.seed(1120)
torch.manual_seed(1120)
torch.cuda.manual_seed_all(1120)

class snnModel(nn.Module):
    def __init__(self, spike_grad, beta=0.5, threshold=0.5, num_steps=20):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 16, 5)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(16, 64, 3)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

        self.conv3 = nn.Conv2d(64, 64, 3)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

   


for i, (res) in tqdm(enumerate(train_loader), total=625):
    # print(res['spike_tensor'].size())
    # print(res['angular_velocity'].size())
    count = count+1
    # break
    # print(count)
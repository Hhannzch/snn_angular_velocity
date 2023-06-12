import torch
import torch.nn as nn
import numpy as np 
import snntorch as snn
import random
from snntorch import surrogate

data = torch.randn([8, 2, 10, 10])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class cnn(nn.Module):
    def __init__(self, spike_grad, beta=0.5, threshold=0.5, num_steps=20):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 16, 5)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.fc = nn.Linear(576, 3)
        self.num_steps = num_steps

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        current1 = self.conv1(x)
        spk1, mem1 = self.lif1(current1, mem1)

        record = spk1.reshape(8, -1)
        out = self.fc(record)

        return out
    
net = cnn(spike_grad=surrogate.fast_sigmoid(slope=25))
out = net(data)
print(out.size())
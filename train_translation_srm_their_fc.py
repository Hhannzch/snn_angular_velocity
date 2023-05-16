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
from panda_data_utils import *

# ==================== load data =============================

data_set = snnDataset(label='xyz_mid_2')
index_train = list(range(0, 200))
index_test = list(range(200, len(data_set)))
train_ = torch.utils.data.Subset(data_set, index_train)
test_ = torch.utils.data.Subset(data_set, index_test)

train_Data = snnDataset(label='xyz_mid_1') \
            +train_ \
            +snnDataset(label='xyz_mid_4') \
            +snnDataset(label='xyz_slow_fast_3') \
            +snnDataset(label='xyz_slow_fast_2')
train_len_total = len(train_Data)
train_len = int(train_len_total * 0.7)
val_len = train_len_total - train_len
train_dataset, val_dataset = torch.utils.data.random_split(train_Data, [train_len, val_len], generator=torch.manual_seed(1120))

train_Set = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
val_Set = DataLoader(val_dataset, batch_size=16, shuffle=True, drop_last=True)

test_Data = test_
test_Set = DataLoader(test_Data, batch_size=1, shuffle=False)

execute = 'train'
label = "srm_pretrain_translation_their"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(1120)
np.random.seed(1120)
torch.manual_seed(1120)
torch.cuda.manual_seed_all(1120)

dir = os.path.join("output", label)
output_dir = Path(dir)
output_dir.mkdir(parents=True, exist_ok=True)

print(len(train_Set))
print(len(val_Set))

# ==================== network structure =============================
logdir = os.path.join(os.getcwd(), 'logs/test')
config = os.path.join(os.getcwd(), 'test_config.yaml')
write = 'store_true'

configs = getTestConfigs(logdir, config)

device_net= configs['general']['hardware']['gpuDevice']
dtype = configs['general']['model']['dtype']

pre_net = getNetwork(configs['general']['model']['type'],
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
    moveToGPUDevice(net, device_net, dtype)
    log_config.copyModelFile(net)

    return net

pre_net = loadNetFromCheckpoint(pre_net, general_config, log_config)

class snnConvModel_pretrained(nn.Module):
    def __init__(self, net, num_steps=20):
        super().__init__()
        self.net = net
        self.fc = nn.Linear(256, 3)
        self.num_steps = num_steps

    def forward(self, x):
        # x: [bs, ts, 2, 180, 240]
        x = x.permute(0, 2, 3, 4, 1)
        batch_size = x.size()[0]

        out_from_net = self.net(x)
        # out_from_net: [bs, channel (256), x (8), y (12), ts (20)]
        out = out_from_net.permute(0, 4, 1, 2, 3)
        # out: [bs, ts (20), channel (256), x (8), y (12)]
        out = torch.sum(out, dim=3)
        out = torch.sum(out, dim=3)
        out = self.fc(out.reshape(batch_size, self.num_steps, -1))

        return out

# ==================== begin training =============================
net = snnConvModel_pretrained(pre_net)
model_save_path = os.path.join(dir, f"snn_model.pth")

if execute == 'train':
    # model.load_state_dict(torch.load(model_save_path))
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    nepoch = 50
    net = net.train()
    net = net.to(device)

    print_graph = []
    print_graph_val = []
    for epoch in range(nepoch):

        train_losses = []
        print_loss = 0.

        for i, (events, targets) in enumerate(train_Set):
            events = events.to(device)
            targets = targets.to(device)

            batch_size = events.size()[0]

            out = net(events)
            
            loss = criterion(out, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            print_loss += loss.item()

            if (i+1) % 10 == 0:

                with torch.no_grad():
                    count = 0.
                    val_loss = 0.
                    for j, (events_val, targets_val) in enumerate(val_Set):
                        events_val = events_val.to(device)
                        targets_val = targets_val.to(device)

                        output = net(events_val)
                        loss_val = criterion(output, targets_val)
                        val_loss = val_loss + loss_val.item()
                        count = count + 1.
                    val_loss = val_loss / count
                    

                print_msg = "[" + str(epoch+1) + ", "+ str(i+1) + "]" + ", running_loss: " + str(print_loss/10) + ", val_loss: " + str(val_loss)
                print(print_msg)
                print_graph.append(print_loss/10)
                print_graph_val.append(val_loss)
                print_loss = 0.0

        train_loss = np.average(train_losses)
        print_msg = "epoch: " + str(epoch+1) + ", train_loss: " + str(train_loss)
        print(print_msg)
        torch.save(net.state_dict(), model_save_path, _use_new_zipfile_serialization=False)

    plt.figure(1)
    plt.plot(print_graph, color='darkgoldenrod', label='train set')
    plt.plot(print_graph_val, color='slateblue', label='val set')
    plt.title("Training process: loss trendency")
    plt.savefig(os.path.join(dir, f"training_loss.png"))
    plt.close()
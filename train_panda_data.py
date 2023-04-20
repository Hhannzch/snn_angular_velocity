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
import os
from panda_data_utils import *

from model import getNetwork
#from utils import moveToGPUDevice
from utils.gpu import moveToGPUDevice
from config.utils import getTestConfigs


# ==================== load network ========================

logdir = os.path.join(os.getcwd(), 'logs/test')
config = os.path.join(os.getcwd(), 'test_config.yaml')
write = 'store_true'

configs = getTestConfigs(logdir, config)

device_net = configs['general']['hardware']['gpuDevice']
dtype = configs['general']['model']['dtype']

net = getNetwork(configs['general']['model']['type'],
                configs['general']['simulation'])

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
train_len = int(train_len_total * 0.9)
val_len = train_len_total - train_len
train_dataset, val_dataset = torch.utils.data.random_split(train_Data, [train_len, val_len], generator=torch.manual_seed(1120))

train_Set = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
val_Set = DataLoader(val_dataset, batch_size=8, shuffle=True, drop_last=True)

test_Data = train_Data + test_
test_Set = DataLoader(test_Data, batch_size=1, shuffle=False)

execute = 'train'
label = "srm"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(1120)
np.random.seed(1120)
torch.manual_seed(1120)
torch.cuda.manual_seed_all(1120)

dir = os.path.join("output", label)
output_dir = Path(dir)
output_dir.mkdir(parents=True, exist_ok=True)
model_save_path = os.path.join(dir, f"snn_model.pth")

print(f"length of the train set is {len(train_Set)}")
print(f"length of the validate set is {len(val_Set)}")

# ==================== begin training =============================

if execute == 'train':
    # model.load_state_dict(torch.load(model_save_path))
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    nepoch = 100
    moveToGPUDevice(net, device_net, dtype)
    net = net.train()

    print_graph = []
    for epoch in range(nepoch):

        if epoch>25:
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

        train_losses = []
        print_loss = 0.

        for i, (events, targets) in enumerate(train_Set):
            events = events.to(device)
            targets = targets.to(device)

            batch_size = events.size()[0]

            events = events.permute(0, 2, 3, 4, 1)
            targets = targets.permute(0, 2, 1)

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
                        events_val = events_val.permute(0, 2, 3, 4, 1)
                        targets_val = targets_val.permute(0, 2, 1)
                        output = net(events_val)
                        loss_val = criterion(output, targets_val)
                        val_loss = val_loss + loss_val.item()
                        count = count + 1.
                    val_loss = val_loss / count
                    

                print_msg = "[" + str(epoch+1) + ", "+ str(i+1) + "]" + ", running_loss: " + str(print_loss/10) + ", val_loss: " + str(val_loss)
                print(print_msg)
                print_graph.append(print_loss/10)
                print_loss = 0.0

        train_loss = np.average(train_losses)
        print_msg = "epoch: " + str(epoch+1) + ", train_loss: " + str(train_loss)
        print(print_msg)
        torch.save(net.state_dict(), model_save_path, _use_new_zipfile_serialization=False)

    plt.figure(1)
    plt.plot(print_graph, color='darkgoldenrod')
    plt.title("Training process: loss trendency")
    plt.savefig(os.path.join(dir, f"training_loss.png"))
    plt.close()

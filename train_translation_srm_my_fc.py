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
import datetime
from model.metric import medianRelativeError_translation, rmse_translation

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
label = "srm_pretrain_translation_my_(test)"
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

# general_config = configs['general']
# log_config = configs['log']

# def loadNetFromCheckpoint(net, general_config, log_config):
#     ckpt_file = general_config['model']['CkptFile']
#     print('Loading checkpoint from {}'.format(ckpt_file))
#     assert ckpt_file
#     checkpoint = torch.load(ckpt_file,
#             map_location=general_config['hardware']['gpuDevice'])

#     net = getNetwork(general_config['model']['type'],
#             general_config['simulation'])
#     net.load_state_dict(checkpoint['model_state_dict'])
#     moveToGPUDevice(net, device_net, dtype)
#     log_config.copyModelFile(net)

#     return net

# pre_net = loadNetFromCheckpoint(pre_net, general_config, log_config)

class snnConvModel_pretrained(nn.Module):
    def __init__(self, net, num_steps=20):
        super().__init__()
        self.net = net
        self.fc = nn.Linear(384, 3)
        self.num_steps = num_steps

    def forward(self, x):
        # x: [bs, ts, 2, 180, 240]
        x = x.permute(0, 2, 3, 4, 1)
        batch_size = x.size()[0]

        out_from_net = self.net(x)
        # out_from_net: [bs, channel (256), x (8), y (12), ts (20)]
        out = out_from_net.permute(0, 4, 1, 2, 3)
        # out: [bs, ts (20), channel (256), x (8), y (12)]
        out1 = out.narrow(2, 0, 64) # [bs, ts, 64, 8, 12]
        out2 = out.narrow(2, 64, 64)
        out3 = out.narrow(2, 128, 64)
        out4 = out.narrow(2, 192, 64)

        out1 = torch.sum(out1, dim=2) / 64
        out2 = torch.sum(out2, dim=2) / 64
        out3 = torch.sum(out3, dim=2) / 64
        out4 = torch.sum(out4, dim=2) / 64

        out = torch.cat((out1, out2, out3, out4), dim=2)
        # out: [bs, ts (20), 4, x, y]

        out = self.fc(out.reshape(batch_size, self.num_steps, -1))

        return out

# ==================== begin training =============================
net = snnConvModel_pretrained(pre_net)
model_save_path = os.path.join(dir, f"snn_model.pth")
log_train_save_path = os.path.join(dir, f"log_train.txt")
log_val_save_path = os.path.join(dir, f"log_val.txt")

if execute == 'train':
    time_before = datetime.datetime.now()
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

    time_after = datetime.datetime.now()

    plt.figure(1)
    plt.plot(print_graph, color='darkgoldenrod', label='train set')
    plt.plot(print_graph_val, color='slateblue', label='val set')
    plt.title("Training process: loss trendency")
    plt.savefig(os.path.join(dir, f"training_loss.png"))
    plt.close()

    print(f"Total training time is: {time_after-time_before}")

    with open(log_train_save_path, "w") as f:
        for i in print_graph:
            f.write(str(i)+'\n')
    with open(log_val_save_path, "w") as f:
        for i in print_graph_val:
            f.write(str(i)+'\n')

elif execute=="test":
    net.load_state_dict(torch.load(model_save_path, map_location=device))
    moveToGPUDevice(net, device_net, dtype)

    changes = []
    changes_hat = []

    loss = [0., 0., 0.]

    with torch.no_grad():
        for i, (event, target) in enumerate(test_Set):
            event = event.to(device)
            target = target.to(device)
            
            change = target.cpu().numpy()
            changes.append(list(change[0][-1]))

            change_hat = net(event)
            
            # change_hat = torch.sum(change_hat, dim=1)
            change_hat = change_hat.cpu().numpy()[0][-1]
            changes_hat.append(list(change_hat))

            loss += abs(change - change_hat)

    print(f"The final test loss is: {loss}")

    changes = np.array(changes)
    changes_hat = np.array(changes_hat)
    print(changes.shape)
    print(changes_hat.shape)

    changes_tensor = torch.tensor(changes)
    changes_hat_tensor = torch.tensor(changes_hat)

    relative_loss = torch.div(torch.abs(changes_tensor - changes_hat_tensor), torch.abs(changes_tensor))
    plt.figure(figsize=(19, 24))
    plt.subplot(311)
    plt.title("x axis")
    plt.plot(relative_loss[:,0].numpy())
    plt.ylim((0, 100))
    plt.subplot(312)
    plt.title("y axis")
    plt.plot(relative_loss[:,1].numpy())
    plt.ylim((0, 100))
    plt.subplot(313)
    plt.title("z axis")
    plt.plot(relative_loss[:,2].numpy())
    plt.ylim((0, 1000))

    plt.savefig(os.path.join(dir, f"relative_error.png"))
    plt.close()

    # print trajectory
    plt.figure(figsize=(19, 24))
    plt.subplot(311)
    plt.title('Results comparison: x-change')
    plt.plot(changes[:,0], color='brown', label='changes')
    plt.plot(changes_hat[:,0], color='royalblue', label='changes_hat', alpha=0.7)
    plt.ylim((-12, 12))
    plt.legend()

    plt.subplot(312)
    plt.title('Results comparison: y-change')
    plt.plot(changes[:,1], color='brown', label='changes')
    plt.plot(changes_hat[:,1], color='royalblue', label='changes_hat', alpha=0.7)
    plt.ylim((-12, 12))
    plt.legend()

    plt.subplot(313)
    plt.title('Results comparison: z-change')
    plt.plot(changes[:,2], color='brown', label='changes')
    plt.plot(changes_hat[:,2], color='royalblue', label='changes_hat', alpha=0.7)
    plt.ylim((-12, 12))
    plt.legend()


    plt.savefig(os.path.join(dir, f"result_position.png"))
    plt.close()

    print('RMSE: {} cm/s'.format(rmse_translation(changes_hat, changes)))
    print('median of relative error: {}'.format(medianRelativeError_translation(changes_hat, changes)))
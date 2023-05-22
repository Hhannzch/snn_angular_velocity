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
import datetime
from model import getNetwork
#from utils import moveToGPUDevice
from utils.gpu import moveToGPUDevice
from config.utils import getTestConfigs


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

execute = 'test'
label = "lif_translation"
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

class snnConvModel(nn.Module):
    def __init__(self, spike_grad, beta=0.5, threshold=0.5, num_steps=20):
        super().__init__()
        
        self.pool = nn.functional.max_pool2d

        self.conv1 = nn.Conv2d(2, 16, 5)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(16, 64, 5)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

        self.conv3 = nn.Conv2d(64, 64, 3)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

        self.fc = nn.Linear(2014, 3)

        self.filter = snn.Leaky(beta=0, threshold=0, spike_grad=spike_grad)
        
        self.num_steps = num_steps

    
    def forward(self, x):
        x = x.transpose(0, 1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        batch_size = x.size()[1]
        channels_size = x.size()[2]
        x_size = x.size()[3]
        y_size = x.size()[4]

        record = []

        for num_step in range(self.num_steps):
            current1 = self.pool(self.conv1(x[num_step]),4)
            spk1, mem1 = self.lif1(current1, mem1)
            current2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(current2, mem2)
            current3 = self.conv3(spk2)
            spk3, mem3 = self.lif3(current3, mem3)

            record.append(mem3)

        record = torch.stack(record)
        record = record.transpose(0, 1) # [bs, 20, channels_size, x_size, y_size]

        record = torch.sum(record, dim=2) / channels_size


        out = self.fc(record.reshape(batch_size, self.num_steps, -1))

        return out

# ==================== begin training =============================
net = snnConvModel(spike_grad=surrogate.fast_sigmoid(slope=25))
model_save_path = os.path.join(dir, f"snn_model.pth")
log_train_save_path = os.path.join(dir, f"log_train.txt")
log_val_save_path = os.path.join(dir, f"log_val.txt")

if execute == 'train':
    time_before = datetime.datetime.now()
    # model.load_state_dict(torch.load(model_save_path))
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    nepoch = 50
    net = net.train()
    net = net.to(device)

    print_graph = []
    print_graph_val = []
    for epoch in range(nepoch):

        if epoch>25:
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

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
    net = net.to(device)

    net.eval()
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
            change_hat = change_hat[0].cpu().numpy()[-1]
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
    plt.ylim((0, 60))
    plt.subplot(312)
    plt.title("y axis")
    plt.plot(relative_loss[:,1].numpy())
    plt.ylim((0, 500))
    plt.subplot(313)
    plt.title("z axis")
    plt.plot(relative_loss[:,2].numpy())
    plt.ylim((0, 60))

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
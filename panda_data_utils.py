import torch
import os
import numpy as np
from matplotlib import animation
import torchvision
import tonic

def snn_animation(snn_data, path):
    """
        Save the animation of one event-based data
        Args:
            snn_data: input event-based data, (len, 2, size, size)
            path: save path
    """
    print(snn_data.size())
    ani = tonic.utils.plot_animation(snn_data)
    pw_writer = animation.PillowWriter(fps=20)
    ani.save(path, writer=pw_writer)

class snnDataset(torch.utils.data.Dataset):
    def __init__(self, label="train", target_type="not_full"):
        super().__init__()
        
        # dir = "/home/czhang13/snn_panda/SNN_data"
        # dir = "/home/chuhan/chuhan/SNN_part_panda/SNN_panda_part_data/data"
        dir = "/home/chuhan/chuhan/rotation_work/snn_angular_velocity/panda_data"

        self.transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.CenterCrop([180, 240])
        ])
        self.target_type = target_type

        if label == 'xyz_mid_1':
            self.path = os.path.join(dir, label)
            self.length = 312
        elif label == 'xyz_mid_2':
            self.path = os.path.join(dir, label)
            self.length = 482 # 482
        elif label == 'xyz_mid_4':
            self.path = os.path.join(dir, label)
            self.length = 442
        elif label == 'xyz_slow_fast_2':
            self.path = os.path.join(dir, label)
            self.length = 401
        elif label == 'xyz_slow_fast_3':
            self.path = os.path.join(dir, label)
            self.length = 443

        else:
            self.length = -1
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        event_path = os.path.join(self.path, "event_based_data", f"events{int(index)}.npy")
        target_path = os.path.join(self.path, "targets", f"target{int(index)}.npy")
        event = np.load(event_path)
        event = torch.tensor(event).float()
        event = self.transform(event)
        target = np.load(target_path)
        target = target * 100 
        target = torch.tensor(target).float()

        return event, target
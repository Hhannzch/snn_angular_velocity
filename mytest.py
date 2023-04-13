import h5py
import torch
from pathlib import Path
from data_loader.testing import TestDatabase
import tonic
from matplotlib import animation

def snn_animation(snn_data, path):
    """
        Save the animation of one event-based data
        Args:
            snn_data: input event-based data, (len, 2, size, size)
            path: save path
    """
    print(snn_data.size())
    ani = tonic.utils.plot_animation(snn_data)
    pw_writer = animation.PillowWriter(fps=25)
    ani.save(path, writer=pw_writer)

data = dict()
hf = h5py.File('/home/chuhan/chuhan/rotation_work/snn_angular_velocity/data/test/seq_23.h5', 'r')
data['ev_xy'] = torch.from_numpy(hf['ev_xy'][()])
data['ev_ts_us'] = torch.from_numpy(hf['ev_ts'][()])
data['ev_pol'] = torch.from_numpy(hf['ev_pol'][()])
data['ang_xyz'] = torch.from_numpy(hf['ang_xyz'][()])
data['ang_ts_us'] = torch.from_numpy(hf['ang_ts'][()])

print(data['ev_xy'].size())
print(data['ev_ts_us'].size())
print(data['ev_pol'].size())
print(data['ang_xyz'].size())
print(data['ang_ts_us'].size())

print(data['ang_ts_us'].numel())


subseq_file = '/home/chuhan/chuhan/rotation_work/snn_angular_velocity/data/test/seq_23.h5'
print(int(''.join(filter(str.isdigit, Path(subseq_file).stem))))

data_dir = '/home/chuhan/chuhan/rotation_work/snn_angular_velocity/data'

test_database = TestDatabase(data_dir)
test_loader = torch.utils.data.DataLoader(
                test_database,
                batch_size=1,
                shuffle=False)
for i, (res) in enumerate(test_loader):
    print(res['spike_tensor'].size())
    event = res['spike_tensor'][0].permute(3,0,1,2)[0:50, :, :, :]
    snn_animation(event, "data1.gif")
    break


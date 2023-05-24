import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path
import numpy

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Parameters")

    root_path = "/home/chuhan/chuhan/rotation_work/snn_angular_velocity/output"

    parser.add_argument("--type", type=str, default="rotation")

    args = parser.parse_args()
    print("Here begin:")

    dir = os.path.join(root_path, f"comparison_{args.type}_srm")
    output_dir = Path(dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    my_dir = os.path.join(root_path, f"srm_pretrain_{args.type}_my")
    their_dir = os.path.join(root_path, f"srm_pretrain_{args.type}_their")

    # read my loss
    my_train_loss = numpy.loadtxt(os.path.join(my_dir, "log_train.txt"))
    my_val_loss = numpy.loadtxt(os.path.join(my_dir, "log_val.txt"))

    # read their loss
    their_train_loss = numpy.loadtxt(os.path.join(their_dir, "log_train.txt"))
    their_val_loss = numpy.loadtxt(os.path.join(their_dir, "log_val.txt"))


    plt.figure(figsize=(24, 19))
    plt.plot(my_train_loss[10:], color='red', alpha=0.7, label='my method')
    plt.plot(their_train_loss[10:], color='blue', alpha=0.7, label='their method')
    plt.legend()
    plt.title("Training loss comparison")
    plt.savefig(os.path.join(dir, f"training_loss_comparison.png"))
    plt.close()

    plt.figure(figsize=(24, 19))
    plt.plot(my_val_loss[10:], color='red', alpha=0.7, label='my method')
    plt.plot(their_val_loss[10:], color='blue', alpha=0.7, label='their method')
    plt.legend()
    plt.title("Validate loss comparison")
    plt.savefig(os.path.join(dir, f"val_loss_comparison.png"))
    plt.close()

    plt.figure(figsize=(24, 19))
    plt.plot(my_train_loss, color='red', alpha=0.7, label='my method')
    plt.plot(their_train_loss, color='blue', alpha=0.7, label='their method')
    plt.legend()
    plt.title("Training loss comparison (full)")
    plt.savefig(os.path.join(dir, f"training_loss_comparison_full.png"))
    plt.close()

    plt.figure(figsize=(24, 19))
    plt.plot(my_val_loss, color='red', alpha=0.7, label='my method')
    plt.plot(their_val_loss, color='blue', alpha=0.7, label='their method')
    plt.legend()
    plt.title("Validate loss comparison (full)")
    plt.savefig(os.path.join(dir, f"val_loss_comparison_full.png"))
    plt.close()

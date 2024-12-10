import argparse
import os
import random
import sys
import time

import numpy as np
import torch

# ddp settings
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ArgoData.data_centerline import Argo2Dataset
from models.structure.banet import get_banet
from utils.helper import Logger


# DDP setup
def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def train(
    epoch, config, train_loader, net, loss, post_process, opt, val_loader=None
):
    net.train()

    train_loader.sampler.set_epoch(epoch)

    num_batches = len(train_loader)
    epoch_per_batch = 1.0 / num_batches
    save_iters = int(np.ceil(config["save_freq"] * num_batches))
    display_iters = int(config["display_iters"] / (config["batch_size"]))
    val_iters = int(config["val_iters"] / (config["batch_size"]))

    start_time = time.time()
    metrics = dict()
    for _, data in tqdm(enumerate(train_loader)):
        epoch += epoch_per_batch
        data = dict(data)
        goal, output = net(data)

        loss_out = loss(goal, output, data)
        post_out = post_process(output, data)
        post_process.append(metrics, loss_out, post_out)

        opt.zero_grad()
        loss_out["loss"].backward()
        lr = opt.step(epoch)

        num_iters = int(np.round(epoch * num_batches))
        if num_iters % save_iters == 0 or epoch >= config["num_epochs"]:
            save_ckpt(net, opt, config["save_dir"], epoch)

        if num_iters % display_iters == 0:
            dt = time.time() - start_time
            # metrics = sync(metrics)
            post_process.display(metrics, dt, epoch, lr)
            start_time = time.time()
            metrics = dict()

        if num_iters % val_iters == 0:
            val(config, val_loader, net, loss, post_process, epoch)

        if epoch >= config["num_epochs"]:
            val(val_loader, net, loss, post_process, epoch)
            return


def val(data_loader, net, loss, post_process, epoch):
    net.eval()

    start_time = time.time()
    metrics = dict()
    for _, data in tqdm(enumerate(data_loader)):

        data = dict(data)
        with torch.no_grad():
            goal, output = net(data)
            loss_out = loss_out = loss(goal, output, data)
            post_out = post_process(output, data)
            post_process.append(metrics, loss_out, post_out)

    dt = time.time() - start_time
    post_process.display(metrics, dt, epoch)
    net.train()


def save_ckpt(net, opt, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # state_dict = net.state_dict()                      # one gpu saving
    state_dict = net.module.state_dict()  # ddp saving

    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_name = "%3.3f.ckpt" % epoch
    torch.save(
        {
            "epoch": epoch,
            "state_dict": state_dict,
            "opt_state": opt.opt.state_dict(),
        },
        os.path.join(save_dir, save_name),
    )


def main(rank, args):

    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.set_device(rank)
    ddp_setup(rank, 2)  # assume 2 gpus are use

    config, collate_fn, net, loss, post_process, opt = get_banet()

    net = DDP(net, device_ids=[rank], find_unused_parameters=True)

    save_dir = config["save_dir"]
    log = os.path.join(save_dir, "log")

    save_dir = config["save_dir"]
    log = os.path.join(save_dir, "log")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sys.stdout = Logger(log)

    train_data = Argo2Dataset(root=args.root, split="train")
    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=DistributedSampler(train_data),
    )

    val_data = Argo2Dataset(root=args.root, split="val")

    val_loader = DataLoader(
        val_data,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=DistributedSampler(val_data),
    )

    epoch = config["epoch"]
    remaining_epochs = int(np.ceil(config["num_epochs"] - epoch))

    for i in range(remaining_epochs):
        train(
            epoch + i,
            config,
            train_loader,
            net,
            loss,
            post_process,
            opt,
            val_loader,
        )

    destroy_process_group()  # destroy the process group


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../")
    # parser.add_argument("--rank", type = int)
    world_size = torch.cuda.device_count()
    args = parser.parse_args()
    mp.spawn(main, args=(args,), nprocs=world_size)
    # main(args)

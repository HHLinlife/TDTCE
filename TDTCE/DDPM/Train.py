import datetime
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets import (GaussianDiffusion, UNet, generate_cosine_schedule,
                  generate_linear_schedule)
from utils.callbacks import LossHistory
from utils.dataloader import Diffusion_dataset_collate, DiffusionDataset
from utils.utils import get_lr_scheduler, set_optimizer_lr, show_config
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":

    Cuda = True
    # ---------------------------------------------------------------------#
    #   Distributed is used to specify whether to use single machine and multi-card distributed operation.
    #   Terminal command only supports Ubuntu. CUDA_VISIBLE_DEVICES is used to specify a graphics card under Ubuntu.
    #   Under Windows system, all graphics cards are called in DP mode by default, and DDP is not supported.
    #   DP mode：
    #       setting            distributed = False
    #       input    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP mode：
    #       setting            distributed = True
    #       input    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    # ---------------------------------------------------------------------#
    distributed = False
    # ---------------------------------------------------------------------#
    #   fp16 is used to specify whether to use mixed precision training.
    #   It can reduce the memory by about half, and needs more than pytorch1.7.1.
    # ---------------------------------------------------------------------#
    fp16 = False
    # ---------------------------------------------------------------------#
    #   If you want to continue practicing breakpoint, set model_path to the trained weight file.
    #   If you want the model to train from 0, set model_path =''.
    # ---------------------------------------------------------------------#
    diffusion_model_path = ""
    channel = 128
    schedule = "linear"
    num_timesteps = 1000
    schedule_low = 1e-4
    schedule_high = 0.02
    input_shape = (16, 16)
    Init_Epoch = 0
    Epoch = 1000
    batch_size = 16
    Init_lr = 2e-4
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = "cos"
    save_period = 25
    save_dir = 'logs'
    num_workers = 4
    annotation_path = "train_traffic.txt"

    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

    if schedule == "cosine":
        betas = generate_cosine_schedule(num_timesteps)
    else:
        betas = generate_linear_schedule(
            num_timesteps,
            schedule_low * 1000 / num_timesteps,
            schedule_high * 1000 / num_timesteps,
        )

    #   Diffusion Network

    diffusion_model = GaussianDiffusion(UNet(3, channel), input_shape, 3, betas=betas)

    if diffusion_model_path != '':
        model_dict = diffusion_model.state_dict()
        pretrained_dict = torch.load(diffusion_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        diffusion_model.load_state_dict(model_dict)


    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, [diffusion_model], input_shape=input_shape)
    else:
        loss_history = None


    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    diffusion_model_train = diffusion_model.train()

    if Cuda:
        if distributed:
            diffusion_model_train = diffusion_model_train.cuda(local_rank)
            diffusion_model_train = torch.nn.parallel.DistributedDataParallel(diffusion_model_train,
                                                                              device_ids=[local_rank],
                                                                              find_unused_parameters=True)
        else:
            cudnn.benchmark = True
            diffusion_model_train = torch.nn.DataParallel(diffusion_model)
            diffusion_model_train = diffusion_model_train.cuda()

    with open(annotation_path) as f:
        lines = f.readlines()
    num_train = len(lines)

    if local_rank == 0:
        show_config(
            input_shape=input_shape, Init_Epoch=Init_Epoch, Epoch=Epoch, batch_size=batch_size, \
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type, \
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train
        )

    if True:

        optimizer = {
            'adam': optim.Adam(diffusion_model_train.parameters(), lr=Init_lr, betas=(momentum, 0.999),
                               weight_decay=weight_decay),
            'adamw': optim.AdamW(diffusion_model_train.parameters(), lr=Init_lr, betas=(momentum, 0.999),
                                 weight_decay=weight_decay),
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, Epoch)

        epoch_step = num_train // batch_size
        if epoch_step == 0:
            raise ValueError("The data set is too small for training. Please expand the data set.")

        train_dataset = DiffusionDataset(lines, input_shape)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=Diffusion_dataset_collate, sampler=train_sampler)

        #   Begin to Train

        for epoch in range(Init_Epoch, Epoch):

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(diffusion_model_train, diffusion_model, loss_history, optimizer,
                          epoch, epoch_step, gen, Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

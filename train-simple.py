import argparse
from tqdm import tqdm, trange
from torch.optim import Adam, AdamW
from torch_geometric.loader import DataLoader

from src.models import *
from src.dataset import * 
import numpy as np
import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.transforms import ToSparseTensor, VirtualNode, ToUndirected
from torch_geometric.nn import to_hetero
import yaml
import os
import csv
import logging

import time

def mean_squared_error_loss(out, batch, **kwargs):
    return torch.mean(torch.square(out- batch.y))

def sum_squared_error_loss(out, batch, **kwargs):
    return torch.sum(torch.square(out - batch.y))

def absolute_error(out, batch, **kwargs):
    return torch.sum(torch.abs(out - batch.y))

def mean_absolute_error(out, batch, **kwargs):
    return torch.mean(torch.abs(out-batch.y))

# def l1_regularization_term(model):
#     l1_regularization = 0.

#     for param in model.parameters():
#         l1_regularization += param.abs().sum()
#     return l1_regularization

def reg_term(model, p):
    reg = 0.
    for param in model.parameters():
        reg += torch.pow(param.abs(), p).sum()
    return reg



def l1_regularized_loss(out, batch, gnn=None, eta=0.1):
    return torch.sum(torch.square(out - batch.y)), eta * reg_term(gnn, 1)

def l0_regularized_loss(out, batch, gnn=None, eta=0.1):
    assert gnn.module_list[0].l0 == True
    l0 = 0.
    for layer in gnn.module_list:
        l0 += layer.get_l0_regularization_term()

    return torch.sum(torch.square(out - batch.y)), eta * l0

def l05_regularized_loss(out, batch, gnn=None, eta=0.1):
    return torch.sum(torch.square(out - batch.y)), eta * reg_term(gnn, 0.5)

def train_simple(train_dataloader, 
                 cfg,
                 epochs=100, 
                 loss_func='mean_squared_error_loss', 
                 device='cuda:0', 
                 log_dir = '/data/sam',
                 lr=0.001,
                 save_freq=100,
                 init='random-init',
                 layer="SimpleBFLayer", 
                 eta=0.1):
    print(cfg)
    if loss_func == 'l0_regularized_loss':
        gnn = globals()[layer](l0_regularizer=True, **cfg)
    else:
        gnn = globals()[layer](**cfg)
    # initialize weights
    if init == 'random-init':
        gnn.random_init()
    elif init == 'random-positive-init':
        gnn.random_positive_init()

    gnn = gnn.to(device)
    optimizer = AdamW(gnn.parameters(), lr=lr)
    record_dir = os.path.join(log_dir, 'record/')

    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    # initialize tensorboard writer
    writer = SummaryWriter(log_dir=record_dir)

    # Logging training information
    log_file = os.path.join(record_dir, 'training_log.log')
    logging.basicConfig(level=logging.INFO, filename=log_file)
    logging.info(f'loss function:{loss_func}')
    logging.info(f'device:{device}')
    logging.info(f'learning rate:{lr}')
    logging.info("GNN model configuration:")
    logging.info(gnn)

    path = os.path.join(record_dir, 'initial_model.pt')
    torch.save({'gnn_state_dict': gnn.state_dict()},
                path)

    for epoch in trange(epochs):
        loss_per_epoch = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch.to(device)
            out = gnn(batch.x, batch.edge_index, batch.edge_attr)
            if 'regularized' in loss_func:
                mse_loss, reg_loss = globals()[loss_func](out, batch, gnn=gnn, eta=eta)
                loss = mse_loss+reg_loss
            else:
                loss = globals()[loss_func](out, batch, gnn=gnn, eta=eta)

            loss_per_epoch = loss.detach()
            loss.backward()
            optimizer.step()

        if 'regularized' in loss_func:
            writer.add_scalar('train/reg_loss', reg_loss.detach(), epoch)
            writer.add_scalar('train/mse_loss', mse_loss.detach(), epoch)
        else:
            writer.add_scalar('train/training_loss', loss_per_epoch, epoch)

        if epoch % save_freq == 0:
            path = os.path.join(record_dir, f'model_{epoch}.pt')
            torch.save({'gnn_state_dict': gnn.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                        path)
    path = os.path.join(log_dir, 'final_model.pt')
    torch.save(gnn.state_dict(), path)
    return gnn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--device', type=str)
    parser.add_argument('--log-dir', type=str)
    parser.add_argument('--num-trials', type=int)
    parser.add_argument('--lr', type=float, nargs='+')
    parser.add_argument('--init', type=str, default='random-init')
    parser.add_argument('--loss-func', type=str, default='mean_squared_error_loss')
    parser.add_argument('--eta', type=float, nargs='+')
    parser.add_argument('--model-configs', type=str)
    parser.add_argument('--layer-type', type=str, default='BFModel')
    parser.add_argument('--perturb', type=int)
    parser.add_argument('--dataset-sizes', type=int, nargs='+')
    parser.add_argument('--search-eta-lr', type=int)

    args = parser.parse_args()

    ## Get model configuration
    with open(args.model_configs, 'r') as file:
        model_configs = yaml.safe_load(file)

    perturb = True if args.perturb == 1 else False
    if args.loss_func != 'mean_squared_error_loss':
        pairs = []
        for eta in args.eta:
            for lr in args.lr:
                pairs.append((lr, eta))
    else:
        pairs = []
        for lr in args.lr:
            pairs.append((lr, 0))

    for lr_eta_pair in pairs:
        lr = lr_eta_pair[0]
        eta = lr_eta_pair[1]
        for model in model_configs:
            cfg = model_configs[model]
            if cfg['bias']:
                bias_for_log_dir='bias'
            else:
                bias_for_log_dir='no-bias'
            activation = cfg['act']
            dataset_sizes = args.dataset_sizes
            for sz in dataset_sizes:
                if args.layer_type == 'SingleSkipBFModel':
                    K = cfg['depth']
                    size = 2 * (K * (K + 1)/2) + 2 * K 
                else:
                    size = sz
                dstr = f'ds-{size}-per' if perturb else f'ds-{size}'
                if 'regularized' in args.loss_func:
                    log_dir = os.path.join(args.log_dir,
                                            args.layer_type,
                                            activation,
                                            args.loss_func,
                                            f'lr-{lr}',
                                            f'eta-{eta}', 
                                            args.init,
                                            model,
                                            bias_for_log_dir,
                                            dstr )
                else:
                    log_dir = os.path.join(args.log_dir,
                                            args.layer_type,
                                            activation,
                                            args.loss_func,
                                            f'lr-{lr}',  
                                            args.init,
                                            model,
                                            bias_for_log_dir,
                                            dstr )
                f = []
                for (dirpath, dirnames, filenames) in os.walk(log_dir):
                    f.extend([int(name) for name in dirnames])
                    break
                cur_trial = 0
                if len(f) > 0:
                    cur_trial = max(f) + 1

                for i in range(cur_trial, cur_trial + args.num_trials):
                    trial = str(i)
                    record_dir = os.path.join(log_dir, trial)
                    
                    if not os.path.exists(record_dir):
                        os.makedirs(record_dir)
                    print("Logging to: ",  record_dir)
                    if args.layer_type == 'SingleSkipBFModel':
                        K = cfg['depth']
                        dataset = construct_ktrain_dataset(K, sz=sz)
                        print('size of dataset:', len(dataset))
                    else:
                        dataset = construct_small_graph_dataset(sz, inject_non_zero=perturb)
                    save_dataset = os.path.join(log_dir, 'train_dataset.pt')
                    torch.save(dataset, save_dataset)
                    #dataset = construct_m_path_dataset(depth + 1, sz=args.dataset_size)
                    print("Size of dataset", len(dataset))
                    batch_sz = len(dataset)
                    train_dataloader = DataLoader(dataset, batch_size = batch_sz, shuffle=True)

                    train_simple(train_dataloader=train_dataloader,
                                cfg=cfg,
                                epochs=args.epochs,
                                loss_func = args.loss_func,
                                device=args.device,
                                log_dir=record_dir,
                                lr=lr,
                                save_freq=args.epochs//2000,
                                init = args.init, 
                                layer=args.layer_type, 
                                eta=eta)

if __name__=='__main__':
    main()
    
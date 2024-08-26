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

def l1_regularization_term(model):
    l1_regularization = 0.

    for param in model.parameters():
        l1_regularization += param.abs().sum()
    return l1_regularization

def l1_regularized_loss(out, batch, gnn=None, eta=0.1):
    return torch.sum(torch.square(out - batch.y)) + eta * l1_regularization_term(gnn)

def l0_regularized_loss(out, batch, gnn=None, eta=0.1):
    assert gnn.module_list[0].l0 == True
    l0 = 0.
    for layer in gnn.module_list:
        l0 += layer.get_l0_regularization_term()

    return torch.sum(torch.square(out - batch.y)) + eta * l0

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
            loss = globals()[loss_func](out, batch, gnn=gnn, eta=eta)

            loss_per_epoch = loss.detach()
            loss.backward()
            optimizer.step()


        writer.add_scalar('train/mse_loss', loss_per_epoch, epoch)

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
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--init', type=str, default='random-init')
    parser.add_argument('--loss-func', type=str, default='mean_squared_error_loss')
    parser.add_argument('--dataset-size', type=str)
    parser.add_argument('--eta', type=float, default=0.10)
    parser.add_argument('--model-configs', type=str)
    parser.add_argument('--layer-type', type=str, default='BFModel')
    parser.add_argument('--perturb', type=int)

    args = parser.parse_args()

    ## Get model configuration
    with open(args.model_configs, 'r') as file:
        model_configs = yaml.safe_load(file)

    perturb = True if args.perturb == 1 else False

    for model in model_configs:
        cfg = model_configs[model]
        if cfg['bias']:
            bias_for_log_dir='bias'
        else:
            bias_for_log_dir='no-bias'
        activation = cfg['act']
        width = cfg['width']
        depth = cfg['depth']
        #dataset_sizes = [4, 8, 16, 32, 64]
        dataset_sizes = [64]
        for sz in dataset_sizes:
            for i in range(args.num_trials):
                trial = str(i)
                dstr = f'ds-{sz}-per' if perturb else f'ds-{sz}'
                if 'regularized' in args.loss_func:
                    log_dir = os.path.join(args.log_dir,
                                           args.layer_type,
                                           activation,
                                           args.loss_func,
                                           f'eta-{args.eta}', 
                                           args.init,
                                           model,
                                           bias_for_log_dir,
                                           dstr, 
                                           trial)
                else:
                    log_dir = os.path.join(args.log_dir,
                                            args.layer_type,
                                            activation,
                                            args.loss_func, 
                                            args.init,
                                            model,
                                            bias_for_log_dir,
                                            dstr, 
                                            trial)

                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                print("Logging to: ",  log_dir)

                dataset = construct_small_graph_dataset(sz, inject_non_zero=perturb)
                if perturb:
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
                            log_dir=log_dir,
                            lr=args.lr,
                            save_freq=50,
                            init = args.init, 
                            layer=args.layer_type, 
                            eta=args.eta)

if __name__=='__main__':
    main()
    
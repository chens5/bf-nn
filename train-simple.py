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

def l1_regularization_term(model):
    l1_regularization = 0.

    for param in model.parameters():
        l1_regularization += param.abs().sum()
    return l1_regularization

def l1_regularized_loss(out, batch, gnn=None, eta=0.1):
    return torch.sum(torch.square(out - batch.y)) + eta * l1_regularization_term(gnn)

def train_simple(train_dataloader, 
                 epochs=100, 
                 loss_func='mean_squared_error_loss', 
                 device='cuda:0', 
                 log_dir = '/data/sam',
                 lr=0.001,
                 save_freq=50):
    
    gnn = SimpleBFLayer()
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

    for epoch in trange(epochs):
        loss_per_epoch = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch.to(device)
            out = gnn(batch.x, batch.edge_index, batch.edge_attr)
            loss = globals()[loss_func](out, batch, gnn=gnn)

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
    parser.add_argument('--trial', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--loss-func', type=str, default='mean_squared_error_loss')

    args = parser.parse_args()
    log_dir = os.path.join(args.log_dir, args.trial)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    dataset = construct_small_eight_graphs()
    train_dataloader = DataLoader(dataset, batch_size = 8)

    train_simple(train_dataloader=train_dataloader,
                 epochs=args.epochs,
                 loss_func = args.loss_func,
                 device=args.device,
                 log_dir=log_dir,
                 lr=args.lr,
                 save_freq=1)

if __name__=='__main__':
    main()
    
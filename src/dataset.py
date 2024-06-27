from typing import Optional
import numpy as np
import torch, queue
from torch import Tensor
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.typing import OptTensor
from tqdm import tqdm, trange
import multiprocessing as mp
import time
import os

import argparse

class BellmanFordStep(Data):
    def __init__(self, 
                 x: Tensor | None = None, 
                 edge_index: OptTensor = None, 
                 edge_attr: OptTensor = None, 
                 y: OptTensor = None, 
                 pos: OptTensor = None, **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)

def two_node_graph(edge_val, beta=100):
    x = torch.tensor([[0], [beta]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 0, 1], 
                               [1, 0, 0, 1]])
    edge_attr = torch.unsqueeze(torch.tensor([edge_val, edge_val, 0, 0], dtype=torch.float), dim=1)
    y = torch.tensor([[0], [edge_val]], dtype=torch.float)
    return x, edge_index, edge_attr, y

def three_node_path(edge_val, beta = 100):
    x = torch.tensor([[0], [edge_val], [beta]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 1, 2, 0, 1, 2], 
                               [1, 0, 2, 1, 0, 1, 2]])
    edge_attr = torch.unsqueeze(torch.tensor([edge_val, edge_val, 0, 0, 0, 0, 0], dtype=torch.float), dim=1)
    y = torch.tensor([[0], [edge_val], [edge_val]], dtype=torch.float)
    return x, edge_index, edge_attr, y

def construct_small_eight_graphs():
    dataset = []
    # Four two node graphs
    for i in range(4):
        x, edge_index, edge_attr, y = two_node_graph(i + 1, beta = 100)
        data = BellmanFordStep(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        dataset.append(data)
    
    # Four 3 node path graphs
    for i in range(4):
        x, edge_index, edge_attr, y = two_node_graph(i + 1, beta = 100)
        data = BellmanFordStep(x = x, edge_index = edge_index, edge_attr = edge_attr, y=y)
        dataset.append(data)

    return dataset
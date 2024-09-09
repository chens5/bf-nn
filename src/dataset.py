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
import copy

import argparse
import itertools

class BellmanFordStep(Data):
    def __init__(self, 
                 x: Tensor | None = None, 
                 edge_index: OptTensor = None, 
                 edge_attr: OptTensor = None, 
                 y: OptTensor = None, 
                 pos: OptTensor = None, **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)

# parameters: G: networkx graph, m: final step, start: first step
def m_step_bf_instance(G, m, start=0, start_node=0):
    G = copy.deepcopy(G)
    temp = {}
    for node in G.nodes:
        temp[node] = {"attr": G.nodes[node]['attr']}
    if start == 0:
        start_dict = copy.deepcopy(temp)
    for k in range(m):
        for node in G.nodes:
            if node == start_node:
                continue
            min_val = G.nodes[node]['attr']
            for neighbor in G.neighbors(node):
                val = G[node][neighbor]['weight'] + G.nodes[neighbor]['attr']
                if val < min_val:
                    min_val = val
            temp[node]['attr'] = min_val
        if k == start - 1:
            start_dict = copy.deepcopy(temp)
        nx.set_node_attributes(G, temp)
    return temp, start_dict

def construct_k_path(k, edge_weights, s=0):
    G = nx.path_graph(k)
    for i in range(len(edge_weights)):
        G[i][i + 1]['weight'] = edge_weights[i]
    nx.set_node_attributes(G, values=1000, name='attr')
    G.nodes[s]['attr'] = 0.0
    return G

def pyg_to_nx(pyg_graph):
    raise NotImplementedError("implementation coming...")

def nx_to_bf_instance(G, m, start=0):
    edge_attr = []
    edge_index = [[], []]
    init_node_features = np.zeros(len(G.nodes))
    final_node_features = np.zeros(len(G.nodes))

    final_bf_m_attrs, start_bf_m_attrs = m_step_bf_instance(G, m, start=start)

    for e in G.edges:
        edge_index[0].append(e[0])
        edge_index[1].append(e[1])

        edge_index[0].append(e[1])
        edge_index[1].append(e[0])

        edge_attr.append(G[e[0]][e[1]]['weight'])
        edge_attr.append(G[e[0]][e[1]]['weight'])
    
    for node in G.nodes:
        init_node_features[node] = start_bf_m_attrs[node]['attr']
        final_node_features[node] = final_bf_m_attrs[node]['attr']
        edge_index[0].append(node)
        edge_index[1].append(node)
        edge_attr.append(0)
    
    data = BellmanFordStep(x = torch.tensor(init_node_features, dtype=torch.float).unsqueeze(1), 
                           y = torch.tensor(final_node_features, dtype=torch.float).unsqueeze(1), 
                           edge_index = torch.tensor(edge_index), 
                           edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1))

    return data
    

def construct_small_graph_dataset(size, inject_non_zero = False):
    dataset = []
    for i in range(size):
        G = construct_k_path(2, [i + 1], s=0)
        data = nx_to_bf_instance(G, 1, start=0)
        dataset.append(data)
    
    for i in range(size):
        perturbation = 0.0
        if inject_non_zero:
            perturbation =np.abs( np.random.normal(loc=0.0, scale=1.0))
        G = construct_k_path(3, [i + 1, perturbation], s=0)
        data = nx_to_bf_instance(G, 2, start=1)
        dataset.append(data)
    return dataset

def construct_m_path_dataset(m, sz=5):
    dataset = []
    for k in range(1, m):
        if k > 1:
            combinations = list(itertools.combinations(range(4*k + 1), 2))
       # sz = len(combinations) if k > 1 else 2
        sz = len(combinations) if k > 1 else 5
        for i in range(sz):
            edge_weight = np.zeros(k)
            if k == 1:
                edge_weight[k - 1] = i
            
            else:
                edge_weight[0] = combinations[i][0]
                edge_weight[k - 1] = combinations[i][1]
            G = construct_k_path(k + 1, edge_weight, s=0)
            data = nx_to_bf_instance(G, k, start = 0)
            dataset.append(data)
    return dataset

def construct_random_path_dataset(m, start_node=0, end=1, start = 0, sz=10):
    dataset = []
    for i in range(sz):
        edge_weight = np.random.uniform(low=1.0, high=10.0, size=m)
        G = construct_k_path(m+1,edge_weight, s=start_node)
        data = nx_to_bf_instance(G, m=end, start=start)
        dataset.append(data)
    return dataset

def construct_k_cycle(k, edge_weights, s=0):
    G = nx.cycle_graph(k)
    for i in range(len(edge_weights) - 1):
        G[i][i + 1]['weight'] = edge_weights[i]
    G[k - 1][0]['weight'] = edge_weights[k - 1]
    nx.set_node_attributes(G, values=200, name='attr')
    G.nodes[s]['attr'] = 0.0
    return G

def construct_cycle_dataset(m, start_node=0, end=1, start=0, sz=10):
    dataset = []
    for i in range(sz):
        edge_weight = np.random.uniform(low=1.0, high=10.0, size=m)
        G = construct_k_cycle(m,edge_weight, s=start_node)
        data = nx_to_bf_instance(G, m=end, start=start)
        dataset.append(data)
    return dataset

def construct_star_graph(num_edges, edges, s=0):
    G = nx.star_graph(num_edges)
    for i in range(1, num_edges + 1):
        G[0][i]['weight'] = edges[i]
    nx.set_node_attributes(G, values=1000, name='attr')
    G.nodes[s]['attr'] = 0.0
    return G

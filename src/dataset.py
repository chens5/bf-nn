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

START_VAL = 1000

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

def pyg_to_nx(pyg_graph):
    raise NotImplementedError("implementation coming...")

def nx_to_pyg(G):
    edge_attr = []
    edge_index = [[], []]
    init_node_features = np.zeros(len(G.nodes))
    for e in G.edges:
        edge_index[0].append(e[0])
        edge_index[1].append(e[1])

        edge_index[0].append(e[1])
        edge_index[1].append(e[0])

        edge_attr.append(G[e[0]][e[1]]['weight'])
        edge_attr.append(G[e[0]][e[1]]['weight'])

    for node in G.nodes:
        init_node_features[node] = G.nodes[node]['attr']
        edge_index[0].append(node)
        edge_index[1].append(node)
        edge_attr.append(0)
    pyg = Data(x=torch.tensor(init_node_features, dtype=torch.float).unsqueeze(1), 
               edge_index=torch.tensor(edge_index), 
               edge_attr=torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1))
    return pyg

def nx_to_bf_instance(G, m, start=0, start_node=0):
    edge_attr = []
    edge_index = [[], []]
    init_node_features = np.zeros(len(G.nodes))
    final_node_features = np.zeros(len(G.nodes))

    final_bf_m_attrs, start_bf_m_attrs = m_step_bf_instance(G, m, start=start, start_node=start_node)
    
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

# K = depth
def construct_test_dataset(K, steps=1):
    edge_set_weights = np.zeros(K)
    dataset = []
    # Cross edge graphs
    graph = construct_cross(K, edge_set_weights, 0, [1, 1 ], s= 0)
    data = nx_to_bf_instance(graph, m=K, start = steps)
    # paths
    for i in range(K):
        edge_weights = np.zeros(K + 1)
        a, b = np.random.uniform(low=1.0, high=10.0, size=2)
        edge_weights[0] = a
        edge_weights[i + 1] = b
        graph= construct_k_path(K+2, edge_weights, 0)
        data = nx_to_bf_instance(graph, m=K+steps, start=1)
        dataset.append(data)
    
    # single edge
    graph = construct_k_path(2, [1.0], 0)
    data = nx_to_bf_instance(graph, m=1, start = 0)
    dataset.append(data)

    # two edges 
    graph = construct_k_path(3, [1.0, 0.0], 0.0)
    data = nx_to_bf_instance(graph, m=2, start = 1)
    dataset.append(data)
    
    return dataset

    

# Construct dataset consisting single edges and two edges
def path_2_3_dataset(size, inject_non_zero = False, **kwargs):
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

# Extended path dataset (for deeper networks)
def path_extend_dataset(K, size, inject_non_zero=False, **kwargs):
    dataset = []
    for i in range(1, K + 1):
        for _ in range(size):
            weight = np.random.uniform(low=1.0, high=20.0, size = i)
            G = construct_k_path(i + 1, weight, s=0)
            data = nx_to_bf_instance(G, i, start =0)
            dataset.append(data)

            weights = np.random.uniform(low=1.0, high=20.0, size = i  + 1 )
            G = construct_k_path(i + 2, weights, s=0)
            data = nx_to_bf_instance(G, i+1, start=i)
            dataset.append(data)
    weights = np.zeros(K)
    weights[0] = START_VAL
    G = construct_k_path(K + 1, weights, s=0)
    data = nx_to_bf_instance(G, K, start = 0)
    dataset.append(data)
    return dataset

# Another extended path dataset for deeper networks
def path_extend_deep_dataset(K, size, **kwargs):
    dataset = []
    sz1 = K + 1
    sz2 = 2 * K + 1
    for _ in range(size):
        
        weights = np.random.uniform(low=1.0, high=20.0, size=sz1 -1)
        G = construct_k_path(sz1, weights, s = 0)
        data = nx_to_bf_instance(G, K, start=0)
        dataset.append(data)

        weights = np.random.uniform(low=1.0, high=20.0, size=sz2 - 1)
        G = construct_k_path(sz2, weights, s=0)
        data = nx_to_bf_instance(G, 2 * K, start=K)
        dataset.append(data)
    # weights = np.zeros(K)
    # weights[0] = START_VAL
    # G = construct_k_path(K + 1, weights, s=0)
    # data = nx_to_bf_instance(G, K, start = 0)
    # dataset.append(data)
    return dataset

# random graph dataset
def erdos_renyi_dataset(size, p=0.5, min_size=4, max_size=10, include_small=False, **kwargs):
    dataset = []
    for _ in range(size):
        graph_sz = np.random.randint(low=min_size, high=max_size)
        G = construct_ER_graph(graph_sz, p=p) 
        data = nx_to_bf_instance(G, graph_sz, start=graph_sz-1)
        dataset.append(data)
    if include_small:
        small_dataset = path_2_3_dataset(4, inject_non_zero=True)
        dataset.extend(small_dataset)
    return dataset

def construct_full_cross_dataset(path_length, cross_w_1, cross_w_2):
    dataset = []
    for j in range(path_length ):
        edge_weights = np.abs( np.random.normal(loc=0.0, scale=1.0, size=path_length + 1))
        H = construct_cross(path_length , edge_weights, j, cross_w_1)
        H_bar = construct_cross(path_length , edge_weights, j, cross_w_2)
        num_nodes = len(H.nodes)

        data_H  = nx_to_bf_instance(H, num_nodes, start=0)
        dataset.append(data_H)
        #print(data_H.edge_index)

        data_H_bar = nx_to_bf_instance(H_bar, num_nodes, start=0)
        dataset.append(data_H_bar)
        #print(data_H_bar.edge_index)
    return dataset

def construct_ktrain_extended(K, size, **kwargs):
    ktrain_dataset = construct_ktrain_dataset(K)
    extra_graphs = path_extend_deep_dataset(K, size)
    return ktrain_dataset+extra_graphs

# K = depth
def construct_ktrain_dataset(K, steps=1, **kwargs):
    edge_set_weights = np.zeros(K)
    dataset = []
    # Cross edge graphs
    graph = construct_cross(K, edge_set_weights, 0, [1, 1 ], s= 0)
    data = nx_to_bf_instance(graph, m=K + 1, start = 1)
    # paths
    for i in range(K):
        for a in range(2*K):
            for b in [0, 2*K + 1]:
                edge_weights = np.zeros(K + 1)
                edge_weights[0] = a
                edge_weights[i + 1] = b
                graph = construct_k_path(K + 2, edge_weights, 0)
                data = nx_to_bf_instance(graph, m = K + 1, start = 1)
                dataset.append(data)
    # single edge
    graph = construct_k_path(2, [1.0], 0)
    data = nx_to_bf_instance(graph, m=1, start = 0)
    dataset.append(data)

    # two edges 
    graph = construct_k_path(3, [1.0, 0.0], 0.0)
    data = nx_to_bf_instance(graph, m=2, start = 1)
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



def construct_cycle_dataset(m, start_node=0, end=1, start=0, sz=10):
    dataset = []
    for i in range(sz):
        edge_weight = np.random.uniform(low=1.0, high=10.0, size=m)
        G = construct_k_cycle(m,edge_weight, s=start_node)
        data = nx_to_bf_instance(G, m=end, start=start)
        dataset.append(data)
    return dataset


### Graph constructors

def construct_cross(num_edges, edges, cross_idx, cross_edge_weights, s=0):
    G = nx.Graph()
    for i in range(num_edges):
        G.add_edge(i, i+1, weight = edges[i])
        G.add_edge(num_edges+i + 1 , num_edges+i + 2, weight=edges[i])
    
    for i in range(num_edges):
        G.add_edge(i, num_edges + 1 +i+1, weight=1.0)
        G.add_edge(i + 1, num_edges + 1 + i, weight=1.0)
    nx.set_node_attributes(G, values=START_VAL, name='attr')
    G.nodes[s]['attr'] = 0.0
    return G

def construct_two_paths(num_edges, edges, s=0):
    G = nx.Graph()
    for i in range(num_edges):
        G.add_edge(i, i+1, weight = edges[i])
        G.add_edge(num_edges+i , num_edges+i + 1, weight=edges[i])
    nx.set_node_attributes(G, values=START_VAL, name='attr')
    G.nodes[s]['attr'] = 0.0
    return G

def construct_star_graph(num_edges, edges, s=0):
    G = nx.star_graph(num_edges)
    for i in range(1, num_edges + 1):
        G[0][i]['weight'] = edges[i]
    nx.set_node_attributes(G, values=START_VAL, name='attr')
    G.nodes[s]['attr'] = 0.0
    return G

def construct_complete_graph(num_nodes,low=1.0, high=2.0, s=0):
    G = nx.complete_graph(num_nodes)
    for e in G.edges:
        v1 = e[0]
        v2 = e[1]
        G[v1][v2]['weight'] = np.random.uniform(low=low, high=high)
    nx.set_node_attributes(G, values=START_VAL, name='attr')
    G.nodes[s]['attr'] = 0.0
    return G

def construct_k_cycle(k, edge_weights, s=0):
    G = nx.cycle_graph(k)
    for i in range(len(edge_weights) - 1):
        G[i][i + 1]['weight'] = edge_weights[i]
    G[k - 1][0]['weight'] = edge_weights[k - 1]
    nx.set_node_attributes(G, values=START_VAL, name='attr')
    G.nodes[s]['attr'] = 0.0
    return G

def construct_k_path(k, edge_weights, s=0):
    G = nx.path_graph(k)
    for i in range(len(edge_weights)):
        G[i][i + 1]['weight'] = edge_weights[i]
    nx.set_node_attributes(G, values=START_VAL, name='attr')
    G.nodes[s]['attr'] = 0.0
    return G

def construct_ER_graph(k, p=0.5, s=0, low=1.0, high=20.0):
    G = nx.erdos_renyi_graph(k, p)
    #num_edges = len(G.edges)
    for edge in G.edges:
        v1 = edge[0]
        v2 = edge[1]
        G[v1][v2]['weight'] = np.random.uniform(low=low, high=high)
    nx.set_node_attributes(G, values=START_VAL, name='attr')
    G.nodes[s]['attr'] = 0.0
    return G
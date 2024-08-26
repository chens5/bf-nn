from src.models import *
from src.dataset import * 
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import os
import yaml

def get_models(top_level_directory, 
                model_configs, 
                dataset_sizes, 
                num_trials, 
                l0=False, 
                perturbation=False):
    model_dictionary = {}
    for modelname in model_configs:
        model_dictionary[modelname] = {}
        for ds in dataset_sizes:
            model_dictionary[modelname][ds] = []
            for t in range(num_trials):
                bstr = f'bias/ds-{ds}-perturbed' if perturbation else f'bias/ds-{ds}'
                final_model_pth = os.path.join(top_level_directory, 
                                               modelname, 
                                               bstr, 
                                               str(t), 
                                               'final_model.pt')
                model_state = torch.load(final_model_pth, map_location='cpu')
                model = BFModel(l0_regularizer=l0, **model_configs[modelname])
                model.load_state_dict(model_state)
                model_dictionary[modelname][ds].append(model)
    return model_dictionary

def single_edge_generalization(model, bound=10):
    dataset = []
    for i in range(1, bound):
        G = construct_k_path(2, [i], s=0)
        data = nx_to_bf_instance(G, 1, start=0)
        dataset.append(data)
    output = []
    for data in dataset:
        out = model(data.x, data.edge_index, data.edge_attr).detach().numpy()
        val = out[1][0]
        output.append(val)
    return np.array(output)


def get_model_error(model, dataset):
    maes = []
    mres = []
    output = []
    for data in dataset:
        out = model(data.x, data.edge_index, data.edge_attr).detach()
        output.append(out)
        diff_out = torch.abs(out - data.y)
        nz = torch.nonzero(data.y.squeeze())[0]
        mre_per_graph = torch.mean(diff_out[nz]/data.y[nz]).item()
        mae_per_graph = torch.mean(torch.abs(out-data.y)).item()
        maes.append(mae_per_graph)
        mres.append(mre_per_graph)
    return maes, mres, output
import torch
from torch.nn import Linear, Parameter, ReLU, LeakyReLU
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import copy

def initialize_mlp(input, hidden, output, layers, activation='relu', **kwargs):
    if layers == 1:
        hidden=output
    if activation == 'relu':
        func = nn.ReLU
    elif activation =='lrelu':
        func = nn.LeakyReLU
    elif activation=='sigmoid':
        func = nn.Sigmoid
    elif activation =='softplus':
        func = nn.Softplus
    else:
        raise NameError('Not implemented')

    phi_layers= []
    phi_layers.append(nn.Linear(input, hidden))
    phi_layers.append(func())
    
    for i in range(layers - 1):
        if i < layers - 2:
            phi_layers.append(nn.Linear(hidden, hidden))
            phi_layers.append(func())
        else:
            phi_layers.append(nn.Linear(hidden, output))

    phi = nn.Sequential(*phi_layers)
    return phi

class L0Linear(nn.Module):
    def __init__(self, in_features, out_features, epsilon_param=1.0, bias=True):
        super(L0Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.include_bias = bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        self.mu_weight = nn.Parameter(torch.normal(mean=torch.zeros((out_features, in_features)), 
                                                   std=torch.ones((out_features, in_features))))
        self.mu_bias = nn.Parameter(torch.normal(mean=torch.zeros(out_features), 
                                                 std=torch.ones(out_features)))
        self.z_weight = None
        self.z_bias = None
        self.epsilon_param = epsilon_param
    
    def get_l0_regularization_term(self):
        return torch.sum(self.z_weight) + torch.sum(self.z_bias)

    def forward(self, x):
        if self.training:
            epsilon_weight = torch.normal(mean=torch.zeros((self.out_features, self.in_features)), 
                                          std = self.epsilon_param  * torch.ones((self.out_features, self.in_features)))
            epsilon_weight = epsilon_weight.to(self.mu_weight.device)
            epsilon_bias = torch.normal(mean=torch.zeros((self.out_features)), 
                                        std = self.epsilon_param * torch.ones((self.out_features)))
            epsilon_bias = epsilon_bias.to(self.mu_weight.device)
        else: 
            epsilon_weight = torch.zeros((self.width, 2))
        self.z_weight = nn.functional.relu(torch.minimum(self.mu_weight + epsilon_weight, torch.ones((self.out_features, self.in_features), device = self.mu_weight.device)))
        self.z_bias = nn.functional.relu(torch.minimum(self.mu_bias + epsilon_bias, torch.ones(self.out_features, device=self.mu_bias.device)))
        out = torch.matmul(x, (self.weight * self.z_weight).T) + (self.bias * self.z_bias)
        return out

class SimpleBFLayer(MessagePassing):
    def __init__(self, act='relu', **kwargs):
        super().__init__(aggr='min')
        self.W_1 = Linear(2, 1)
        if act == 'relu':
            self.relu = ReLU()
        elif act == 'leaky_relu':
            self.relu = LeakyReLU(negative_slope=0.01)
        self.w_2 = Parameter(torch.rand(1))
        self.b_2 = Parameter(torch.rand(1))
    
    # Random initialization sampling from N(0, 1)
    def random_init(self):
        torch.nn.init.normal_(self.W_1.weight, 0.0 , 1.0)
        torch.nn.init.normal_(self.w_2, 0.0, 1.0)
        torch.nn.init.normal_(self.b_2, 0.0, 1.0)
        torch.nn.init.normal_(self.W_1.bias, 0.0, 1.0)

    # random positive initialization (sampling from positive uniform)
    def random_positive_init(self):
        torch.nn.init.uniform_(self.W_1.weight, a = 0.0, b = 2.0)
        torch.nn.init.uniform_(self.w_2, a = 0.0, b = 2.0)
        torch.nn.init.uniform_(self.b_2, a = 0.0, b = 2.0)
        torch.nn.init.uniform_(self.W_1.bias, a = 0.0, b = 2.0)

    def forward(self, x, edge_index, edge_attr):        
        return self.relu(self.w_2 * self.propagate(edge_index, x=x, edge_attr=edge_attr) + self.b_2)

    def message(self, x_j, edge_attr):
        return self.relu(self.W_1(torch.cat((x_j, edge_attr), dim=-1)))

class SingleLayerArbitraryWidthBFLayer(MessagePassing):
    def __init__(self, 
                 width=2,
                 out_features = 1,
                 in_features = 1, 
                 bias=False, 
                 act='ReLU', 
                 l0_regularizer=False, 
                 **kwargs):
        super().__init__(aggr='min')
        self.l0 = l0_regularizer
        # Add 1 to input features to account for edge attribute 
        if l0_regularizer:
            self.W_1 = L0Linear(in_features= in_features + 1, out_features=width, bias=bias)
            self.W_2 = L0Linear(in_features=width, out_features=out_features, bias=bias)
        else:
            self.W_1 = Linear(in_features = in_features + 1, out_features = width, bias=bias)
            self.W_2 = Linear(in_features=width, out_features=out_features, bias=bias)

        # self.act = ReLU()
        self.act = globals()[act]()

        if bias:
            self.bias = True
        else:
            self.bias = False
    
    def random_init(self):
        torch.nn.init.normal_(self.W_1.weight, 0.0, 1.0)
        
        torch.nn.init.normal_(self.W_2.weight, 0.0, 1.0)
        
        if self.bias:
            torch.nn.init.normal_(self.W_1.bias, 0.0, 1.0)
            torch.nn.init.normal_(self.W_2.bias, 0.0, 1.0)

    
    def random_positive_init(self):
        torch.nn.init.uniform_(self.W_1.weight, a = 0.0, b=1.0)        
        torch.nn.init.uniform_(self.W_2.weight, a = 0.0, b=1.0)
        
        if self.bias:
            torch.nn.init.uniform_(self.W_1.bias, a = 0.0, b=1.0)
            torch.nn.init.uniform_(self.W_2.bias, a = 0.0, b=1.0)
    
    def get_l0_regularization_term(self):
        assert isinstance(self.W_1, L0Linear)
        assert isinstance(self.W_2, L0Linear)

        return self.W_1.get_l0_regularization_term() + self.W_2.get_l0_regularization_term()

    def forward(self, x, edge_index, edge_attr):
        return self.act(self.W_2(self.propagate(edge_index, x=x, edge_attr=edge_attr)))
    
    def message(self, x_j, edge_attr):
        return self.act(self.W_1(torch.cat((x_j, edge_attr), dim=-1)))



class BFModel(nn.Module):
    def __init__(self, 
                 width, 
                 depth, 
                 bias=True, 
                 act='ReLU', 
                 l0_regularizer = False, 
                 **kwargs):
        super(BFModel, self).__init__()
        self.module_list = nn.ModuleList([SingleLayerArbitraryWidthBFLayer(width, bias=bias, act=act, l0_regularizer=l0_regularizer) for _ in range(depth)])

    def random_positive_init(self):
        for layer in self.module_list:
            layer.random_positive_init()
    
    def random_init(self):
        for layer in self.module_list:
            layer.random_init()

    def forward(self, x, edge_index, edge_attr):
        for layer in self.module_list:            
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return x

class SingleSkipLayer(MessagePassing):
    def __init__(self, aggregation_config, update_config, act = 'ReLU' , **kwargs ):
        super(SingleSkipLayer, self).__init__()

        self.aggregation_layer = initialize_mlp(**aggregation_config)
         
        self.update_layer = initialize_mlp(**update_config)
        self.act = globals()[act]()

    def forward(self, x, edge_index, edge_attr):
        from_aggregation = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        update_ = torch.cat((from_aggregation, x), dim=1)

        return self.act(self.update_layer(self.propagate(edge_index, x=x, edge_attr=edge_attr)))
    
    def message(self, x_j, edge_attr):
        return self.act(self.aggregation_layer(torch.cat((x_j, edge_attr), dim=-1)))

class SingleSkipBFModel(nn.Module):
    def __init__(self, 
                 initial_aggregation_config, 
                 aggregation_config,
                 update_config, 
                 depth, 
                 bias=True, 
                 act='ReLU', 
                 l0_regularizer = False, 
                 **kwargs):
        super(SingleSkipBFModel, self).__init__()
        # Safety checks
        # (1) check that input for update MLP accounts for residual 
        assert 2 * aggregation_config['output'] == update_config['input'] and 2 * initial_aggregation_config['output'] == update_config['input']
        # account for edge features
        assert initial_aggregation_config['input'] == 2 
        assert aggregation_config['input'] == update_config['output'] + 1 

        final_update_cfg = copy.deepcopy(update_config)
        final_update_cfg['output'] = 1
        lst = []
        lst.append(SingleSkipLayer(initial_aggregation_config, update_config))
        for i in range(depth - 2): 
            lst.append(SingleSkipLayer(aggregation_config, update_config))
        lst.append(SingleSkipLayer(aggregation_config, final_update_cfg))
        self.module_list = nn.ModuleList(*lst)
    
    def forward(self, x, edge_index, edge_attr):
        for layer in self.module_list:            
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return x
import torch
from torch.nn import Linear, Parameter, ReLU, LeakyReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

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
    def __init__(self, width=2, bias=False, act='relu'):
        super().__init__(aggr='min')
        self.W_1 = Linear(in_features = 2, out_features = width, bias=bias)

        self.act = ReLU()

        self.W_2 = Linear(in_features=width, out_features=1, bias=bias)
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

    def forward(self, x, edge_index, edge_attr):
        return self.W_2(self.propagate(edge_index, x=x, edge_attr=edge_attr))
    
    def message(self, x_j, edge_attr):
        return self.act(self.W_1(torch.cat((x_j, edge_attr), dim=-1)))


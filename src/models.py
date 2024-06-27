import torch
from torch.nn import Linear, Parameter, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class SimpleBFLayer(MessagePassing):
    def __init__(self):
        super().__init__(aggr='min')
        self.W_1 = Linear(2, 1)
        self.relu = ReLU()
        self.w_2 = Parameter(torch.rand(1))
        self.b_2 = Parameter(torch.rand(1))

    def forward(self, x, edge_index, edge_attr):        
        return self.relu(self.w_2 * self.propagate(edge_index, x=x, edge_attr=edge_attr) + self.b_2)

    def message(self, x_j, edge_attr):

        return self.relu(self.W_1(torch.cat((x_j, edge_attr), dim=-1)))
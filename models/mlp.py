import torch
import torch.nn as nn
import torch.nn.functional as F

# MLP with lienar output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, act='sigmoid'):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True # default is linear model
        self.num_layers = num_layers
        self.act = torch.relu if act == 'relu' else torch.sigmoid

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.act(self.linear(x))
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.dropout(F.relu(self.linears[layer](h)),0.6, training=self.training)  # 0.6
            return self.act(self.linears[self.num_layers - 1](h))


class MLP_fun(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP_fun, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return torch.sigmoid(self.linear(x))
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.dropout(F.relu(self.linears[layer](h)), 0.5, training=self.training)
            return torch.relu(self.linears[self.num_layers - 1](h))
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, HeteroConv


class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, activation_fn=torch.nn.ReLU(), dropout_p=0.2,
                 gnn_type='gcn'):
        super().__init__()
        if gnn_type == 'gcn':
            self.gnn_block = GCNConv
        elif gnn_type == 'gat':
            self.gnn_block = GATConv
        elif gnn_type == 'sage':
            self.gnn_block = SAGEConv
        else:
            raise Exception(f'{gnn_type} is not supported.')

        self.conv1 = self.gnn_block(num_node_features, hidden_dim)
        if num_classes > 2:
            self.conv2 = self.gnn_block(hidden_dim, num_classes)
            self.output_fn = torch.nn.LogSoftmax(dim=1)
        else:
            self.conv2 = self.gnn_block(hidden_dim, 1)
            self.output_fn = torch.nn.LogSigmoid()
        self.activation_fn = activation_fn
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, node_features, edge_index):
        node_features = self.conv1(node_features, edge_index)
        node_features = self.activation_fn(node_features)
        node_features = self.dropout(node_features)
        node_features = self.conv2(node_features, edge_index)
        return torch.exp(self.output_fn(node_features))


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, dropout_p=0.2, **kwargs):
        super().__init__(**kwargs)

        self.out_dim = 1

        # Features
        self.in_dim = in_dim

        # Define the layers
        self.layers = nn.ModuleList([
            nn.Linear(self.in_dim, hidden_dim),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dim)
        ])

        # Output layer
        self.output_fn = nn.Sigmoid()

        # Custom loss component
        self.loss_fn = torch.nn.BCELoss(reduction="mean")

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x) -> torch.FloatTensor:
        for layer in self.layers:
            x = layer(x)
        return self.output_fn(x)

    def loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

    def params(self):
        for name, param in self.named_parameters():
            yield param
            

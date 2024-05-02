import torch
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


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

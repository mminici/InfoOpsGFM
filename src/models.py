import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, HeteroConv, GraphConv


class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, activation_fn=torch.nn.ReLU(), dropout_p=0.2,
                 gnn_type='gcn'):
        super().__init__()
        if gnn_type == 'gcn':
            self.gnn_block = GCNConv
        elif gnn_type == 'gat':
            self.gnn_block = GATConv
        elif gnn_type == 'sage':
            # self.gnn_block = SAGEConv
            self.gnn_block = lambda x, y: GraphConv(x, y, aggr='mean')
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

    def forward(self, node_features, edge_index, edge_weight=None):
        if edge_weight is None:
            node_features = self.conv1(node_features, edge_index)
        else:
            node_features = self.conv1(node_features, edge_index, edge_weight)
        node_features = self.activation_fn(node_features)
        node_features = self.dropout(node_features)
        if edge_weight is None:
            node_features = self.conv2(node_features, edge_index)
        else:
            node_features = self.conv2(node_features, edge_index, edge_weight)
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


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        self.scale = self.head_dim ** 0.5

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size = x.size(0)

        # Linear projections
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = self.softmax(attn_weights)

        # Weighted sum of values
        out = torch.matmul(attn_weights, v)

        # Concatenate heads and pass through the final linear layer
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        out = self.out_linear(out)

        return out


class HeteroGNN(torch.nn.Module):
    def __init__(self, gnn_type, num_node_features, hidden_dim, edge_types, activation_fn=torch.nn.ReLU(), dropout_p=0.2,
                 num_classes=2, aggr_fn='sum'):
        super(HeteroGNN, self).__init__()
        if gnn_type == 'gcn':
            self.gnn_block = GCNConv
        elif gnn_type == 'gat':
            self.gnn_block = GATConv
        elif gnn_type == 'sage':
            self.gnn_block = SAGEConv
        else:
            raise Exception(f'{gnn_type} is not supported.')
        self.conv1 = HeteroConv({('node', edge_type, 'node'): self.gnn_block(num_node_features, hidden_dim) for edge_type in edge_types}, aggr=aggr_fn)
        self.activation_fn = activation_fn
        self.dropout = torch.nn.Dropout(dropout_p)
        if aggr_fn == 'cat':
            multiplier = 5
        else:
            multiplier = 1
        if num_classes > 2:
            self.conv2 = HeteroConv(
                {('node', edge_type, 'node'): self.gnn_block(multiplier * hidden_dim, num_classes) for edge_type in
                 edge_types}, aggr=aggr_fn)
            self.output_fn = torch.nn.LogSoftmax(dim=1)
        else:
            self.conv2 = HeteroConv({('node', edge_type, 'node'): self.gnn_block(multiplier * hidden_dim, 1)
                                     for edge_type in edge_types}, aggr=aggr_fn)
            # self.projection1 = SelfAttention(multiplier, 5)
            # self.projection2 = nn.Linear(multiplier, 1)
            self.output_fn = torch.nn.LogSigmoid()

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: self.activation_fn(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        # x_dict = {key: torch.exp(self.output_fn(self.projection2(self.projection1(x)))) for key, x in x_dict.items()}
        x_dict = {key: torch.exp(self.output_fn(x)) for key, x in x_dict.items()}
        return x_dict



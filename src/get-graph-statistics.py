import pathlib
import torch
import networkx as nx
import numpy as np

from my_utils import get_edge_index_from_networkx
from torch_geometric.utils import homophily

# Hyper parameters
traces_list = ['coRT']
dataset_name = 'UAE_sample'

base_dir = pathlib.Path.cwd().parent
processed_data_dir = base_dir / 'data' / 'processed' / dataset_name
# Save the obtained network on disk
traces_fname = "".join(traces_list)
network = nx.read_graphml(processed_data_dir / f'network{traces_fname}.gml')
node_labels = np.load(processed_data_dir / f'network{traces_fname}_node_labels.npy')

edge_index = get_edge_index_from_networkx(network)
node_labels = torch.LongTensor(node_labels)

node_homophily = homophily(edge_index, node_labels, method='node')
edge_homophily = homophily(edge_index, node_labels, method='edge')
ci_edge_homophily = homophily(edge_index, node_labels, method='edge_insensitive')

print(f'Dataset: {dataset_name}; Traces: {traces_fname}')
print(f'Number of nodes: {network.number_of_nodes()}, number of edges: {network.number_of_edges()}')
print(f'Prevalence of IO drivers: {round((100*node_labels.sum()).item()/node_labels.shape[0], 2)}')
print(f'Node Homophily: {node_homophily}')
print(f'Edge Homophily: {edge_homophily}')
print(f'Edge Insensitive Homophily: {ci_edge_homophily}')

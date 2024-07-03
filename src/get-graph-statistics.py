import pathlib
import torch
import numpy as np
import pickle

from my_utils import get_edge_index_from_networkx
from torch_geometric.utils import homophily

# Hyper parameters
dataset_name = 'cuba'
filter_th = 0.99

base_dir = pathlib.Path.cwd().parent
processed_data_dir = base_dir / 'data' / 'processed' / dataset_name
with open(processed_data_dir / f'{filter_th}_datasets.pkl', 'rb') as file:
    datasets = pickle.load(file)

# Print fused network statistics
network = datasets['graph']
fusedNet_nodes_list = np.array(list(network.nodes()))
node_labels = torch.tensor(datasets['labels']).long()
fusedNet_io_drivers = fusedNet_nodes_list[datasets['labels'] == 1]
edge_index = get_edge_index_from_networkx(network)

node_homophily = homophily(edge_index, node_labels, method='node')
edge_homophily = homophily(edge_index, node_labels, method='edge')
ci_edge_homophily = homophily(edge_index, node_labels, method='edge_insensitive')

number_of_excluded_users = datasets['excluded_users'].userid.nunique()
tot_number_of_io_users = node_labels.sum().item() + number_of_excluded_users

print(f'Dataset: {dataset_name}; Traces: ALL')
print(f'Number of nodes: {network.number_of_nodes()}, number of edges: {network.number_of_edges()}')
print('Num IO drivers:', len(fusedNet_io_drivers))
print(f'Prevalence of IO drivers: {round((100 * node_labels.sum()).item() / node_labels.shape[0], 2)}')
print(f'IO drivers coverage (%): {round((100 * node_labels.sum().item()) / tot_number_of_io_users, 2)}')
print(f'Node Homophily: {node_homophily}')
print(f'Edge Homophily: {edge_homophily}')
print(f'Edge Insensitive Homophily: {ci_edge_homophily}')

for trace_name in ['coRT', 'coURL', 'hashSeq', 'fastRT', 'tweetSim']:
    # Save the obtained network on disk
    network = datasets[trace_name]
    nodes_list = np.array(list(network.nodes()))
    node_labels = torch.tensor(datasets['labels']).long()
    node_labels = node_labels[list(network.nodes())]
    io_drivers = nodes_list[node_labels == 1]
    edge_index = get_edge_index_from_networkx(network)

    node_homophily = homophily(edge_index, node_labels, method='node')
    edge_homophily = homophily(edge_index, node_labels, method='edge')
    ci_edge_homophily = homophily(edge_index, node_labels, method='edge_insensitive')

    print(f'Dataset: {dataset_name}; Traces: {trace_name}')
    print(f'Number of nodes: {network.number_of_nodes()}, number of edges: {network.number_of_edges()}')
    print('Num IO drivers:', len(io_drivers))
    print('Coverage IO drivers (%):', round(100 * len(io_drivers)/tot_number_of_io_users, 2))
    print(f'Prevalence of IO drivers: {round((100 * node_labels.sum()).item() / node_labels.shape[0], 2)}')
    print(f'Node Homophily: {node_homophily}')
    print(f'Edge Homophily: {edge_homophily}')
    print(f'Edge Insensitive Homophily: {ci_edge_homophily}')

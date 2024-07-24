import pathlib
import torch
import numpy as np
import pickle

from my_utils import get_edge_index_from_networkx, remove_edge_attributes
from torch_geometric.utils import homophily

# Hyper parameters
dataset_name = 'UAE_sample'
filter_th = 0.7
print_subnets = True


def remove_isolated_and_self_loop_nodes(graph):
    # Create a list of nodes to be removed
    nodes_to_remove = []

    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        # Check if the node has no neighbors or only a self-loop
        if len(neighbors) == 0 or (len(neighbors) == 1 and neighbors[0] == node):
            nodes_to_remove.append(node)

    # Remove the identified nodes from the graph
    graph.remove_nodes_from(nodes_to_remove)

    return graph


# Function to count isolated nodes including those with self-loops
def count_isolated_nodes_including_self_loops(graph):
    isolated_nodes = []
    for node in graph.nodes():
        # Get all the neighbors of the node
        neighbors = list(graph.neighbors(node))
        # Check if the node has no neighbors or only has a self-loop
        if len(neighbors) == 0 or (len(neighbors) == 1 and neighbors[0] == node):
            isolated_nodes.append(node)
    return len(isolated_nodes)


base_dir = pathlib.Path.cwd().parent
processed_data_dir = base_dir / 'data' / 'processed' / dataset_name
with open(processed_data_dir / f'{filter_th}_datasets.pkl', 'rb') as file:
    datasets = pickle.load(file)

# Print fused network statistics
raw_network = datasets['graph']
fusedNet_nodes_list = np.array(list(raw_network.nodes()))
node_labels = torch.tensor(datasets['labels']).long()
fusedNet_io_drivers = fusedNet_nodes_list[datasets['labels'] == 1]
network = remove_edge_attributes(raw_network)
number_of_excluded_users = count_isolated_nodes_including_self_loops(network)
network = remove_isolated_and_self_loop_nodes(network)
edge_index = get_edge_index_from_networkx(network)

node_homophily = homophily(edge_index, node_labels, method='node')
edge_homophily = homophily(edge_index, node_labels, method='edge')
ci_edge_homophily = homophily(edge_index, node_labels, method='edge_insensitive')

tot_number_of_io_users = node_labels.sum().item() + number_of_excluded_users

print(f'Dataset: {dataset_name}; Traces: ALL')
print(f'Number of nodes: {network.number_of_nodes()}, number of edges: {network.number_of_edges()}')
print(f'Number of isolated nodes: {number_of_excluded_users}')
print('Num IO drivers:', len(fusedNet_io_drivers))
print(f'Prevalence of IO drivers: {round((100 * node_labels.sum()).item() / node_labels.shape[0], 2)}')
print(f'IO drivers coverage (%): {round((100 * node_labels.sum().item()) / tot_number_of_io_users, 2)}')
print(f'Node Homophily: {node_homophily}')
print(f'Edge Homophily: {edge_homophily}')
print(f'Edge Insensitive Homophily: {ci_edge_homophily}')

print()

if print_subnets:
    for trace_name in ['coRT', 'coURL', 'hashSeq', 'fastRT', 'tweetSim']:
        # Save the obtained network on disk
        raw_network = datasets[trace_name]
        network = remove_edge_attributes(raw_network)
        isolated_nodes_number = count_isolated_nodes_including_self_loops(network)
        network = remove_isolated_and_self_loop_nodes(network)
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
        print('Coverage IO drivers (%):', round(100 * len(io_drivers) / tot_number_of_io_users, 2))
        print(f'Prevalence of IO drivers: {round((100 * node_labels.sum()).item() / node_labels.shape[0], 2)}')
        print(f'Node Homophily: {node_homophily}')
        print(f'Edge Homophily: {edge_homophily}')
        print(f'Edge Insensitive Homophily: {ci_edge_homophily}')
        print(f'Number of isolated nodes: {isolated_nodes_number}')


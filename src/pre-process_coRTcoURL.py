import pathlib
import pickle
import pandas as pd
import networkx as nx
import numpy as np

from preprocessing_util import coRetweet, coURL

# Script Hyper-Parameters
DATASET_NAME = 'cuba'
extract_largest_connected_component = True

# Script constants
CONTROL_FILE_IDX, IO_FILE_IDX = 0, 1
filename_dict = {'UAE_sample': ['control_driver_tweets_uae_082019.jsonl', 'uae_082019_tweets_csv_unhashed.csv'],
                 'cuba': ['control_driver_tweets_cuba_082020.jsonl', 'cuba_082020_tweets_csv_unhashed.csv']}

# Execution
# Basic definitions
base_dir = pathlib.Path.cwd().parent
data_dir = base_dir / 'data' / 'raw'
processed_data_dir = base_dir / 'data' / 'processed'

# Read raw data
io_df = pd.read_csv(data_dir / DATASET_NAME / filename_dict[DATASET_NAME][IO_FILE_IDX], sep=",")
control_df = pd.read_json(data_dir / DATASET_NAME / filename_dict[DATASET_NAME][CONTROL_FILE_IDX], lines=True)

# Build Co-Retweet
control, treated = control_df.copy(), io_df.copy()
control_column_names = ['id', 'user', 'retweeted_status']
treated_column_names = ['userid', 'tweetid', 'retweet_tweetid']
control = control[control_column_names]
treated = treated[treated_column_names]
coRetweet_network = coRetweet(control, treated)

# Build Co-URL
control, treated = control_df.copy(), io_df.copy()
control_column_names = ['user', 'entities', 'id']
treated_column_names = ['tweetid', 'userid', 'urls']
control = control[control_column_names]
treated = treated[treated_column_names]
coURL_network = coURL(control, treated)

# Merge the two networks
fused_network = nx.compose(coRetweet_network, coURL_network)
if extract_largest_connected_component:
    # Extract largest connected component
    connected_components_ordered_list = sorted(nx.connected_components(fused_network), key=len, reverse=True)
    fused_network = nx.Graph(fused_network.subgraph(connected_components_ordered_list[0]))
    fused_network.remove_edges_from(nx.selfloop_edges(fused_network))
    print(f'nodes: {fused_network.number_of_nodes()} edges: {fused_network.number_of_edges()}')

# Remap nodes
noderemapping = {nodeid: i for i, nodeid in enumerate(fused_network.nodes())}
noderemapping_rev = {v: k for k, v in noderemapping.items()}
node_labels = np.zeros(len(noderemapping))
all_io_users = set(io_df.userid.astype(str).values.tolist())
for nodeid in noderemapping_rev:
    raw_nodeid = noderemapping_rev[nodeid]
    node_labels[nodeid] = 1 if raw_nodeid in all_io_users else 0

# Save the obtained network on disk
nx.write_graphml(fused_network, processed_data_dir / DATASET_NAME / 'fused_network.gml')
np.save(processed_data_dir / DATASET_NAME / 'fused_network_node_labels.npy', node_labels)
with open(processed_data_dir / DATASET_NAME / 'noderemapping.pkl', 'wb') as file:
    pickle.dump(noderemapping, file)
with open(processed_data_dir / DATASET_NAME / 'noderemapping_rev.pkl', 'wb') as file:
    pickle.dump(noderemapping_rev, file)

import pandas as pd
import numpy as np
import networkx as nx
import pathlib

from tqdm.notebook import tqdm
from scipy import sparse

# Script hyperparameters
DATASET_NAME = 'UAE_sample'
CONTROL_USERS_FILENAME = 'control_users.csv'
IO_USERS_FILENAME = 'io_drivers_users.csv'

# Basic definitions
base_dir = pathlib.Path().cwd().parent
data_dir = base_dir / 'data' / 'raw'
processed_data_dir = base_dir / 'data' / 'processed'

control_df = pd.read_csv(processed_data_dir / DATASET_NAME / CONTROL_USERS_FILENAME, sep=",",
                         lineterminator='\n', index_col=0)
iodrivers_df = pd.read_csv(processed_data_dir / DATASET_NAME / IO_USERS_FILENAME, sep=",",
                           lineterminator='\n', index_col=0)

#### Step 1. Limit control tweets to the same time interval of the Information Campaign
iodrivers_df['tweet_time'] = pd.to_datetime(iodrivers_df.tweet_time)
control_df['tweet_time'] = pd.to_datetime(control_df.tweet_time)

io_start_time, io_end_time = iodrivers_df['tweet_time'].min(), iodrivers_df['tweet_time'].max()
print(f'num raw control tweets: {len(control_df.index)}')

control_df = control_df[control_df['tweet_time'] >= io_start_time]
control_df = control_df[control_df['tweet_time'] <= io_end_time]
print(f'num control tweets after limiting to IO time interval: {len(control_df.index)}')

#### Step 2. Print basic statistics
print(f'[CONTROL] users: {control_df.userid.nunique()} tweets: {control_df.tweetid.nunique()}')
print(f'[IO] users: {iodrivers_df.userid.nunique()} tweets: {iodrivers_df.tweetid.nunique()}')

#### Step 3. Build co-retweet interaction network
# limit the set of tweets to those having at least 1 retweet
retweet_id_set = list()
retweet_id_set += iodrivers_df.retweet_tweetid.unique().tolist()
retweet_id_set += control_df.retweet_tweetid.unique().tolist()
retweet_id_set = list(set(retweet_id_set))
print(f'num tweets with at least a retweet: {len(retweet_id_set)}')

retweet_id_to_idx = {retweet_id_set[i]: i for i in range(len(retweet_id_set))}

# Build one-hot encoding of user retweets history
io_users_retweet_matrix = np.zeros(shape=(iodrivers_df.userid.nunique(), len(retweet_id_set)), dtype=int)
control_users_retweet_matrix = np.zeros(shape=(control_df.userid.nunique(), len(retweet_id_set)), dtype=int)
io_users_set, control_users_set = iodrivers_df.userid.unique(), control_df.userid.unique()

# Populating io matrix
for i in tqdm(range(len(io_users_set)), 'io drivers'):
    user_id = io_users_set[i]
    for retweet_id in iodrivers_df[iodrivers_df.userid == user_id].retweet_tweetid:
        if not pd.isna(retweet_id):
            io_users_retweet_matrix[i, retweet_id_to_idx[retweet_id]] = 1

# Populating control matrix
for i in tqdm(range(len(control_users_set)), 'control users'):
    user_id = control_users_set[i]
    for retweet_id in control_df[control_df.userid == user_id].retweet_tweetid:
        if not pd.isna(retweet_id):
            control_users_retweet_matrix[i, retweet_id_to_idx[retweet_id]] = 1

# convert to np.uint16 for operation efficiency
io_users_retweet_matrix = io_users_retweet_matrix.astype(np.uint16)
control_users_retweet_matrix = control_users_retweet_matrix.astype(np.uint16)
# exclude users having zero retweets
io_num_retweets = io_users_retweet_matrix.sum(1)
control_num_retweets = control_users_retweet_matrix.sum(1)
io_users_retweet_matrix = io_users_retweet_matrix[io_num_retweets > 0]
control_users_retweet_matrix = control_users_retweet_matrix[control_num_retweets > 0]
io_users_set = io_users_set[io_num_retweets > 0]
control_users_set = control_users_set[control_num_retweets > 0]

print('users having at least a retweet')
print(f'[IO] users: {len(io_users_set)}')
print(f'[CONTROL] users: {len(control_users_set)}')

user_retweet_matrix = np.vstack([control_users_retweet_matrix, io_users_retweet_matrix])
# Exclude tweets having zero retweets
num_tweet_was_retweeted = user_retweet_matrix.sum(0)
user_retweet_matrix = user_retweet_matrix[:, num_tweet_was_retweeted > 0]
num_tweet_was_retweeted = num_tweet_was_retweeted[num_tweet_was_retweeted > 0]
print(f'tweets having at least a retweet: {user_retweet_matrix.shape[1]}')

# Convert the matrix to sparse format for matrix multiplication efficiency
row_indexes, col_indexes = np.where(user_retweet_matrix > 0)
values = np.ones(len(row_indexes))
# Normalize the matrix using TF-IDF approach
idf_retweet = np.log(user_retweet_matrix.shape[0] / num_tweet_was_retweeted) + 1
for i in tqdm(range(len(values))):
    values[i] /= idf_retweet[col_indexes[i]]
sparse_user_retweet_matrix = sparse.csr_matrix((values, (row_indexes, col_indexes)))

# Build user-user interaction network based on retweet behavioral trace
shared_retweet_matrix = sparse_user_retweet_matrix @ sparse_user_retweet_matrix.T
retweet_network = nx.from_scipy_sparse_array(shared_retweet_matrix)
print(f'nodes: {retweet_network.number_of_nodes()} edges: {retweet_network.number_of_edges()}')

# Extract largest connected component
connected_components_ordered_list = sorted(nx.connected_components(retweet_network), key=len, reverse=True)
retweet_network_lcc = nx.Graph(retweet_network.subgraph(connected_components_ordered_list[0]))
retweet_network_lcc.remove_edges_from(nx.selfloop_edges(retweet_network_lcc))
print(f'nodes: {retweet_network_lcc.number_of_nodes()} edges: {retweet_network_lcc.number_of_edges()}')

# Extract node labels
nodes_labels = np.ones(user_retweet_matrix.shape[0], dtype=int)
nodes_labels[:control_users_retweet_matrix.shape[0]] = 0
# Limit node labels to those nodes present in the largest connected component
nodes_labels = nodes_labels[retweet_network_lcc.nodes()]
print(f'LCC contains {retweet_network_lcc.number_of_nodes()} nodes and {retweet_network_lcc.number_of_edges()} edges. \
            {nodes_labels.sum()}/{retweet_network_lcc.number_of_nodes()} nodes are IO drivers')

#### Step 4. Save the obtained network on disk
nx.write_graphml(retweet_network_lcc, processed_data_dir / DATASET_NAME / 'lcc_retweet.gml')
np.save(processed_data_dir / DATASET_NAME / 'lcc_retweet_labels.npy', nodes_labels)

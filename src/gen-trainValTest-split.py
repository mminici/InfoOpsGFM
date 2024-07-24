import argparse
import pathlib
import pickle
import networkx as nx
import pandas as pd
import numpy as np

from data_loader import load_dataset
from sklearn.model_selection import train_test_split

DATASET_FILENAME = 'datasets.pkl'
CONTROL_FILE_IDX, IO_FILE_IDX = 0, 1
filename_dict = {'UAE_sample': ['control_driver_tweets_uae_082019.jsonl', 'uae_082019_tweets_csv_unhashed.csv'],
                 'cuba': ['control_driver_tweets_cuba_082020.jsonl', 'cuba_082020_tweets_csv_unhashed.csv']}


def save_dataset(data_dir, data, filter_th, tr_perc, undersampling):
    fname = f'{filter_th}_{DATASET_FILENAME}'
    if tr_perc != 0.6:
        fname = f'{filter_th}_{DATASET_FILENAME}_{tr_perc}'
    fname += f'_{undersampling}U'
    with open(data_dir / fname, 'wb') as file:
        pickle.dump(data, file)


def load_network(path, net_type=None):
    with open(path, 'rb') as f:
        network = pickle.load(f)
        return network


def retain_nodes_in_graph(graph, nodes_to_retain):
    """
    Removes all nodes not present in the nodes_to_retain list from the graph.

    Parameters:
    graph (networkx.Graph): A NetworkX graph.
    nodes_to_retain (list): A list of nodes to be retained in the graph.

    Returns:
    networkx.Graph: The graph with only the specified nodes retained.
    """
    nodes_to_remove = [node for node in graph.nodes if node not in nodes_to_retain]
    graph.remove_nodes_from(nodes_to_remove)
    return graph


def flip_true_entries_stratified(arr, labels, percentage):
    """
    Randomly flip a given percentage of True entries to False in a boolean NumPy array in a stratified way
    based on the given node labels.

    Parameters:
    arr (np.ndarray): A boolean NumPy array.
    labels (np.ndarray): A NumPy array of node labels.
    percentage (float): A float between 0.0 and 1.0 representing the percentage of True entries to flip.

    Returns:
    np.ndarray: A new boolean NumPy array with the specified percentage of True entries flipped to False.
    """
    if not (0.0 <= percentage <= 1.0):
        raise ValueError("Percentage must be between 0.0 and 1.0.")

    # Ensure arr and labels have the same length
    if len(arr) != len(labels):
        raise ValueError("The length of arr and labels must be the same.")

    # Create a copy of the original array to modify
    new_arr = arr.copy()

    # Find unique labels
    unique_labels = np.unique(labels)

    for label in unique_labels:
        # Get indices of entries with the current label
        label_indices = np.where(labels == label)[0]

        # Find the True entries among these indices
        true_indices = np.where(arr[label_indices] == True)[0]
        num_true = len(true_indices)

        # Calculate the number of True entries to flip for this label
        num_to_flip = int(num_true * percentage)

        if num_to_flip > 0:
            # Randomly select indices to flip
            flip_indices = np.random.choice(true_indices, num_to_flip, replace=False)

            # Flip the selected indices in the new array
            new_arr[label_indices[flip_indices]] = False

    return new_arr


def main(dataset_name, train_perc, val_perc, test_perc, tweet_sim_threshold, num_splits, min_tweets, under_sampling):
    assert train_perc + val_perc + test_perc == 1.0, 'Split percentages do not sum to 1.'
    # Basic definitions
    base_dir = pathlib.Path.cwd().parent
    processed_datadir = base_dir / 'data' / 'processed'
    data_dir = processed_datadir / dataset_name
    data_dir.mkdir(parents=True, exist_ok=True)
    if under_sampling is not None:
        # Open the dataset
        dataset = load_dataset(data_dir, tweet_sim_threshold, train_perc, None)
        for run_id in dataset['splits']:
            tr_mask = dataset['splits'][run_id]['train']
            dataset['splits'][run_id]['train'] = flip_true_entries_stratified(tr_mask, dataset['labels'], under_sampling)
        save_dataset(data_dir, dataset, tweet_sim_threshold, train_perc, under_sampling)
        return
    # Load networks
    print('Loading similarity networks...')
    coRT = load_network(data_dir / 'coRT.pkl', net_type='coRT')
    coURL = load_network(data_dir / 'coURL.pkl', net_type='coURL')
    hashSeq = load_network(data_dir / 'hashSeq.pkl', net_type='hashSeq')
    fastRT = load_network(data_dir / 'fastRT.pkl', net_type='fastRT')
    tweetSim = load_network(data_dir / 'tweetSim.pkl', net_type='tweetSim')
    filter_th = 'noFilt'
    if tweet_sim_threshold is not None:
        print('Filtering edges from tweetSim network...')
        # Remove edges from tweetSim
        edges_to_remove = [(u, v) for u, v, w in tweetSim.edges(data=True) if w['weight'] < tweet_sim_threshold]
        tweetSim.remove_edges_from(edges_to_remove)
        filter_th = str(round(tweet_sim_threshold, 2))
    # Fusing the 5 similarity networks
    fusedNet = nx.compose(tweetSim, fastRT)
    fusedNet = nx.compose(fusedNet, hashSeq)
    fusedNet = nx.compose(fusedNet, coURL)
    fusedNet = nx.compose(fusedNet, coRT)
    # Applying node filtering
    # At the moment, we exclude all users having less than 10 tweets
    print(base_dir / 'data' / 'raw' / dataset_name / filename_dict[dataset_name][CONTROL_FILE_IDX])
    control_df = pd.read_json(base_dir / 'data' / 'raw' / dataset_name / filename_dict[dataset_name][CONTROL_FILE_IDX],
                              lines=True)
    control_df['userid'] = control_df['user'].apply(lambda x: np.int64(x['id']))
    print('Importing IO drivers file...')
    print(base_dir / 'data' / 'raw' / dataset_name / filename_dict[dataset_name][IO_FILE_IDX])
    iodrivers_df = pd.read_csv(base_dir / 'data' / 'raw' / dataset_name / filename_dict[dataset_name][IO_FILE_IDX],
                               sep=",")
    # Grouping by 'userid' and filtering those with at least 5 rows
    io_drivers_userids = iodrivers_df.groupby('userid').filter(lambda x: len(x) >= min_tweets)['userid'].unique().astype(str)
    control_userids = control_df.groupby('userid').filter(lambda x: len(x) >= min_tweets)['userid'].unique().astype(str)
    filtered_userids = np.concatenate([io_drivers_userids, control_userids])
    coRT = retain_nodes_in_graph(coRT, filtered_userids)
    coURL = retain_nodes_in_graph(coURL, filtered_userids)
    hashSeq = retain_nodes_in_graph(hashSeq, filtered_userids)
    fastRT = retain_nodes_in_graph(fastRT, filtered_userids)
    tweetSim = retain_nodes_in_graph(tweetSim, filtered_userids)
    fusedNet = retain_nodes_in_graph(fusedNet, filtered_userids)
    # Add a self-loop for each node in the list if it's not already in the graph
    for node_id in io_drivers_userids:
        if not coRT.has_node(node_id):
            coRT.add_edge(node_id, node_id, weight=1.0)
        if not coURL.has_node(node_id):
            coURL.add_edge(node_id, node_id, weight=1.0)
        if not hashSeq.has_node(node_id):
            hashSeq.add_edge(node_id, node_id, weight=1.0)
        if not fastRT.has_node(node_id):
            fastRT.add_edge(node_id, node_id, weight=1.0)
        if not tweetSim.has_node(node_id):
            tweetSim.add_edge(node_id, node_id, weight=1.0)
        if not fusedNet.has_node(node_id):
            fusedNet.add_edge(node_id, node_id, weight=1.0)
    # Remap nodes
    print('Remap nodes of fused network...')
    noderemapping = {nodeid: i for i, nodeid in enumerate(fusedNet.nodes())}
    noderemapping_rev = {v: k for k, v in noderemapping.items()}
    node_labels = np.zeros(len(noderemapping))
    for nodeid in noderemapping_rev:
        raw_nodeid = noderemapping_rev[nodeid]
        node_labels[nodeid] = 1 if raw_nodeid in io_drivers_userids else 0
    # Relabel nodes
    fusedNet = nx.relabel_nodes(fusedNet, noderemapping)
    coRT = nx.relabel_nodes(coRT, noderemapping)
    coURL = nx.relabel_nodes(coURL, noderemapping)
    hashSeq = nx.relabel_nodes(hashSeq, noderemapping)
    fastRT = nx.relabel_nodes(fastRT, noderemapping)
    tweetSim = nx.relabel_nodes(tweetSim, noderemapping)
    datasets = {'graph': fusedNet.copy(), 'coRT': coRT.copy(), 'coURL': coURL.copy(), 'hashSeq': hashSeq.copy(),
                'fastRT': fastRT.copy(), 'tweetSim': tweetSim.copy(), 'noderemapping': noderemapping,
                'noderemapping_rev': noderemapping_rev, 'labels': np.copy(node_labels), 'splits': {}}
    # Perform train-val-test split
    print(f'Performing train-val-test split (tr, val, test: {train_perc}, {val_perc}, {test_perc})...')
    for run_id in range(num_splits):
        x_train, x_test, _, y_test = train_test_split(range(len(node_labels)), node_labels,
                                                      test_size=1 - train_perc,
                                                      stratify=node_labels)
        x_val, x_test, _, _ = train_test_split(x_test, y_test,
                                               test_size=test_perc / (test_perc + val_perc),
                                               stratify=y_test)
        run_train_mask = np.full(fill_value=False, shape=len(node_labels))
        run_train_mask[x_train] = True
        run_val_mask = np.full(fill_value=False, shape=len(node_labels))
        run_val_mask[x_val] = True
        run_test_mask = np.full(fill_value=False, shape=len(node_labels))
        run_test_mask[x_test] = True
        # Store training, val and test masks
        datasets['splits'][run_id] = {'train': np.copy(run_train_mask),
                                      'val': np.copy(run_val_mask),
                                      'test': np.copy(run_test_mask)}
    # Save dataset to be reusable
    save_dataset(data_dir, datasets, filter_th, train_perc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess dataset to produce train-val-test split")
    parser.add_argument('-dataset_name', '--dataset', type=str, help='Dataset', default='UAE_sample')
    parser.add_argument('-train_perc', '--train', type=float, help='Training percentage', default=.6)
    parser.add_argument('-val_perc', '--val', type=float, help='Validation percentage', default=.2)
    parser.add_argument('-test_perc', '--test', type=float, help='Test percentage', default=.2)
    parser.add_argument('-num_splits', '--splits', type=int, help='Num of train-val-test splits', default=5)
    parser.add_argument('-tweet_sim_threshold', '--tsim_th', type=float, help='Threshold over which we retain an edge '
                                                                              'in tweet similarity network',
                        default=.7)
    parser.add_argument('-min_tweets', '--min_tweets', type=int,
                        help='Minimum number of tweets a user needs to have to be included in the dataset',
                        default=10)
    parser.add_argument('-heterogeneous', '--het', action='store_true', help="If True, return all the networks "
                                                                             "otherwise return the fused")
    parser.add_argument('-under_sampling', '--under', type=float, help='undersampling percentage', default=None)
    args = parser.parse_args()
    main(args.dataset, args.train, args.val, args.test, args.tsim_th, args.splits, args.min_tweets, args.under)

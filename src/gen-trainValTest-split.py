import argparse
import pathlib
import pickle
import networkx as nx
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

DATASET_FILENAME = 'datasets.pkl'
CONTROL_FILE_IDX, IO_FILE_IDX = 0, 1
filename_dict = {'UAE_sample': ['control_driver_tweets_uae_082019.jsonl', 'uae_082019_tweets_csv_unhashed.csv'],
                 'cuba': ['control_driver_tweets_cuba_082020.jsonl', 'cuba_082020_tweets_csv_unhashed.csv']}


def save_dataset(data_dir, data, filter_th):
    with open(data_dir / f'{filter_th}_{DATASET_FILENAME}', 'wb') as file:
        pickle.dump(data, file)


def load_network(path, net_type=None):
    with open(path, 'rb') as f:
        network = pickle.load(f)
        return network


def get_all_user_ids_twitter_data(base_dir, dataset_name):
    data_dir = base_dir / 'data' / 'raw'
    treated = pd.read_csv(data_dir / dataset_name / filename_dict[dataset_name][IO_FILE_IDX], sep=",")
    return set(treated.userid.astype(str).values.tolist())


def main(dataset_name, train_perc, val_perc, test_perc, tweet_sim_threshold, num_splits):
    assert train_perc + val_perc + test_perc == 1.0, 'Split percentages do not sum to 1.'
    # Basic definitions
    base_dir = pathlib.Path.cwd().parent
    processed_datadir = base_dir / 'data' / 'processed'
    data_dir = processed_datadir / dataset_name
    data_dir.mkdir(parents=True, exist_ok=True)
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
    user_within_fusedNet = set([np.int64(elem) for elem in fusedNet.nodes()])
    # Remap nodes
    print('Remap nodes of fused network...')
    noderemapping = {nodeid: i for i, nodeid in enumerate(fusedNet.nodes())}
    noderemapping_rev = {v: k for k, v in noderemapping.items()}
    node_labels = np.zeros(len(noderemapping))
    all_io_users = get_all_user_ids_twitter_data(base_dir, dataset_name)
    for nodeid in noderemapping_rev:
        raw_nodeid = noderemapping_rev[nodeid]
        node_labels[nodeid] = 1 if raw_nodeid in all_io_users else 0
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
    # Add to the datasets the dataframes of users not present in the network
    tweets_df = pd.read_csv(base_dir / 'data' / 'processed' / dataset_name / 'IO_mostPop_tweet_texts.csv', index_col=0)
    user_with_tweets = set(tweets_df.userid.unique())
    user_with_tweets_excluded_from_fusedNet = user_with_tweets - user_within_fusedNet
    result_df_limited = tweets_df[tweets_df.userid.isin(user_with_tweets_excluded_from_fusedNet)]
    datasets['excluded_users'] = result_df_limited
    # Save dataset to be reusable
    save_dataset(data_dir, datasets, filter_th)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess dataset to produce train-val-test split")
    parser.add_argument('-dataset_name', '--dataset', type=str, help='Dataset', default='cuba')
    parser.add_argument('-train_perc', '--train', type=float, help='Training percentage', default=.6)
    parser.add_argument('-val_perc', '--val', type=float, help='Validation percentage', default=.2)
    parser.add_argument('-test_perc', '--test', type=float, help='Test percentage', default=.2)
    parser.add_argument('-num_splits', '--splits', type=int, help='Num of train-val-test splits', default=5)
    parser.add_argument('-tweet_sim_threshold', '--tsim_th', type=float, help='Threshold over which we retain an edge '
                                                                              'in tweet similarity network', default=.99)
    parser.add_argument('-heterogeneous', '--het', action='store_true', help="If True, return all the networks "
                                                                             "otherwise return the fused")
    args = parser.parse_args()
    main(args.dataset, args.train, args.val, args.test, args.tsim_th, args.splits)

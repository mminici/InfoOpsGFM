import numpy as np
import networkx as nx
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from preprocessing_util import coRetweet, coURL

DATASET_FILENAME = 'datasets.pkl'
ALLOWED_TRACES = ['coURL', 'coRT']
# Script constants
CONTROL_FILE_IDX, IO_FILE_IDX = 0, 1
filename_dict = {'UAE_sample': ['control_driver_tweets_uae_082019.jsonl', 'uae_082019_tweets_csv_unhashed.csv'],
                 'cuba': ['control_driver_tweets_cuba_082020.jsonl', 'cuba_082020_tweets_csv_unhashed.csv']}


def check_data_exists(data_dir):
    return (data_dir / DATASET_FILENAME).exists()


def save_data(data_dir, data):
    with open(data_dir / DATASET_FILENAME, 'wb') as file:
        pickle.dump(data, file)


def load_data(data_dir):
    with open(data_dir / DATASET_FILENAME, 'rb') as file:
        datasets = pickle.load(file)
    return datasets


def create_data_loader(dataset_name, base_dir, data_dir, hyper_params):
    # Check if data already exists or you want to create it from scratch
    if not check_data_exists(data_dir) or ("overwrite_data" in hyper_params and hyper_params["overwrite_data"]):
        traces_list = hyper_params['traces_list']
        # start with the first trace
        tracename = traces_list[0]
        network = get_twitter_data(base_dir, dataset_name, tracename)
        for tracename in traces_list[1:]:
            new_network = get_twitter_data(base_dir, dataset_name, tracename)
            network = nx.compose(network, new_network)
        if hyper_params['extract_largest_connected_component']:
            # Extract largest connected component
            connected_components_ordered_list = sorted(nx.connected_components(network), key=len,
                                                       reverse=True)
            network = nx.Graph(network.subgraph(connected_components_ordered_list[0]))
            network.remove_edges_from(nx.selfloop_edges(network))
            print(f'nodes: {network.number_of_nodes()} edges: {network.number_of_edges()}')

        # Remap nodes
        noderemapping = {nodeid: i for i, nodeid in enumerate(network.nodes())}
        noderemapping_rev = {v: k for k, v in noderemapping.items()}
        node_labels = np.zeros(len(noderemapping))
        all_io_users = get_all_user_ids_twitter_data(base_dir, dataset_name)
        for nodeid in noderemapping_rev:
            raw_nodeid = noderemapping_rev[nodeid]
            node_labels[nodeid] = 1 if raw_nodeid in all_io_users else 0

        # Save the obtained network on disk
        traces_fname = "".join(traces_list)
        nx.write_graphml(network, base_dir / 'processed' / dataset_name / f'network{traces_fname}.gml')
        np.save(base_dir / 'processed' / dataset_name / f'network{traces_fname}_node_labels.npy', node_labels)
        with open(base_dir / 'processed' / dataset_name / f'noderemapping{traces_fname}.pkl', 'wb') as file:
            pickle.dump(noderemapping, file)
        with open(base_dir / 'processed' / dataset_name / f'noderemapping_rev{traces_fname}.pkl', 'wb') as file:
            pickle.dump(noderemapping_rev, file)

        # Create num_splits random data splits
        datasets = {'graph': network.copy(), 'labels': np.copy(node_labels), 'splits': {}}
        io_drivers_idxs, control_users_idxs = np.where(node_labels == 1)[0], np.where(node_labels == 0)[0]

        if hyper_params['is_few_shot']:
            # Sample test set
            test_mask = np.full(fill_value=True, shape=len(node_labels))
            # Select few shots for test set
            test_io_ids = np.random.choice(range(len(io_drivers_idxs)),
                                           size=int(hyper_params['test_perc'] * len(io_drivers_idxs)), replace=False)
            test_control_ids = np.random.choice(range(len(control_users_idxs)),
                                                size=int(hyper_params['test_perc'] * len(control_users_idxs)),
                                                replace=False)
            test_mask[io_drivers_idxs[test_io_ids]] = True
            test_mask[control_users_idxs[test_control_ids]] = True
            # Exclude users in the test set from training and validation sampling
            io_drivers_idxs_after_test = np.delete(io_drivers_idxs, test_io_ids)
            control_users_idxs_after_test = np.delete(control_users_idxs, test_control_ids)
            for run_id in range(hyper_params['num_splits']):
                io_drivers_idxs = np.copy(io_drivers_idxs_after_test)
                control_users_idxs = np.copy(control_users_idxs_after_test)
                # Select few shots for training
                train_io_ids = np.random.choice(range(len(io_drivers_idxs)),
                                                size=hyper_params['train_few_shot_samples'],
                                                replace=False)
                train_control_ids = np.random.choice(range(len(control_users_idxs)),
                                                     size=hyper_params['train_few_shot_samples'], replace=False)
                run_train_mask = np.full(fill_value=False, shape=len(node_labels))
                run_train_mask[io_drivers_idxs[train_io_ids]] = True
                run_train_mask[control_users_idxs[train_control_ids]] = True
                # Delete training from set from which we will draw validation examples
                io_drivers_idxs = np.delete(io_drivers_idxs, train_io_ids)
                control_users_idxs = np.delete(control_users_idxs, train_control_ids)
                # Select few shots for validation
                val_io_ids = np.random.choice(range(len(io_drivers_idxs)), size=hyper_params['val_few_shot_samples'],
                                              replace=False)
                val_control_ids = np.random.choice(range(len(control_users_idxs)),
                                                   size=hyper_params['val_few_shot_samples'],
                                                   replace=False)
                run_val_mask = np.full(fill_value=False, shape=len(node_labels))
                run_val_mask[io_drivers_idxs[val_io_ids]] = True
                run_val_mask[control_users_idxs[val_control_ids]] = True
                # Store training, val and test masks
                datasets['splits'][run_id] = {'train': np.copy(run_train_mask),
                                              'val': np.copy(run_val_mask),
                                              'test': np.copy(test_mask)}
        else:
            for run_id in range(hyper_params['num_splits']):
                x_train, x_test, _, y_test = train_test_split(range(len(node_labels)), node_labels,
                                                              test_size=1 - hyper_params['train_perc'],
                                                              stratify=True)
                x_val, x_test, _, _ = train_test_split(x_test, y_test,
                                                       test_size=hyper_params['test_perc'] / (
                                                               hyper_params['test_perc'] + hyper_params[
                                                           'val_perc']),
                                                       stratify=True)
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
        save_data(data_dir, datasets)
    else:
        datasets = load_data(data_dir)
    return datasets


def get_twitter_data(base_dir, dataset_name, tracename):
    if tracename in ALLOWED_TRACES:
        if tracename == 'coRT':
            return get_twitter_coRT(base_dir, dataset_name)
        if tracename == 'coURL':
            return get_twitter_coURL(base_dir, dataset_name)
    raise Exception(f'Trace {tracename} not in ALLOWED_TRACES.')


def get_twitter_coRT(base_dir, dataset_name):
    data_dir = base_dir / 'data' / 'raw'
    # Read raw data
    treated = pd.read_csv(data_dir / dataset_name / filename_dict[dataset_name][IO_FILE_IDX], sep=",")
    control = pd.read_json(data_dir / dataset_name / filename_dict[dataset_name][CONTROL_FILE_IDX], lines=True)
    # Build Co-Retweet
    control_column_names = ['id', 'user', 'retweeted_status']
    treated_column_names = ['userid', 'tweetid', 'retweet_tweetid']
    control = control[control_column_names]
    treated = treated[treated_column_names]
    return coRetweet(control, treated)


def get_twitter_coURL(base_dir, dataset_name):
    data_dir = base_dir / 'data' / 'raw'
    # Read raw data
    treated = pd.read_csv(data_dir / dataset_name / filename_dict[dataset_name][IO_FILE_IDX], sep=",")
    control = pd.read_json(data_dir / dataset_name / filename_dict[dataset_name][CONTROL_FILE_IDX], lines=True)
    # Build Co-URL
    control_column_names = ['user', 'entities', 'id']
    treated_column_names = ['tweetid', 'userid', 'urls']
    control = control[control_column_names]
    treated = treated[treated_column_names]
    return coURL(control, treated)


def get_all_user_ids_twitter_data(base_dir, dataset_name):
    data_dir = base_dir / 'data' / 'raw'
    treated = pd.read_csv(data_dir / dataset_name / filename_dict[dataset_name][IO_FILE_IDX], sep=",")
    return set(treated.userid.astype(str).values.tolist())

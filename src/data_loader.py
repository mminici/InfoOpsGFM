import numpy as np
import networkx as nx
import pickle

DATASET_FILENAME = 'datasets.pkl'
RETWEET_GRAPH_FILENAME = 'fused_network.gml'
USER_LABELS_FILENAME = 'fused_network_node_labels.npy'


def check_data_exists(data_dir):
    return (data_dir / DATASET_FILENAME).exists()


def save_data(data_dir, data):
    with open(data_dir / DATASET_FILENAME, 'wb') as file:
        pickle.dump(data, file)


def load_data(data_dir):
    with open(data_dir / DATASET_FILENAME, 'rb') as file:
        signed_datasets = pickle.load(file)
    return signed_datasets


def create_data_loader(dataset_name, base_dir, data_dir, hyper_params, device):
    if not check_data_exists(data_dir) or ("overwrite_data" in hyper_params and hyper_params["overwrite_data"]):
        # Read network and user labels
        retweet_network_lcc = nx.read_graphml(base_dir / 'data' / 'processed' / dataset_name / RETWEET_GRAPH_FILENAME)
        user_labels = np.load(base_dir / 'data' / 'processed' / dataset_name / USER_LABELS_FILENAME)
        # Create num_splits random data splits
        assert "num_splits" in hyper_params, 'num_splits arg is missing'
        datasets = {'graph': retweet_network_lcc.copy(), 'labels': np.copy(user_labels), 'splits': {}}
        # Sample test set
        test_mask = np.full(fill_value=True, shape=len(user_labels))
        io_drivers_idxs, control_users_idxs = np.where(user_labels == 1)[0], np.where(user_labels == 0)[0]
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
            train_io_ids = np.random.choice(range(len(io_drivers_idxs)), size=hyper_params['train_few_shot_samples'],
                                            replace=False)
            train_control_ids = np.random.choice(range(len(control_users_idxs)),
                                                 size=hyper_params['train_few_shot_samples'], replace=False)
            run_train_mask = np.full(fill_value=False, shape=len(user_labels))
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
            run_val_mask = np.full(fill_value=False, shape=len(user_labels))
            run_val_mask[io_drivers_idxs[val_io_ids]] = True
            run_val_mask[control_users_idxs[val_control_ids]] = True
            # Store training, val and test masks
            datasets['splits'][run_id] = {'train': np.copy(run_train_mask), 'val': np.copy(run_val_mask),
                                          'test': np.copy(test_mask)}
        # Save dataset to be reusable
        save_data(data_dir, datasets)
    else:
        datasets = load_data(data_dir)
    return datasets

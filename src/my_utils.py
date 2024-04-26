import random
import torch
import pathlib
import uuid
import numpy as np


def set_seed(seed):
    if seed is None:
        seed = 12121995
    print(f"[ Using Seed : {seed} ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_train_val_test_masks(size, val_perc=0.1, test_perc=0.1):
    train_perc = 1. - val_perc - test_perc
    assert train_perc > 0 and val_perc > 0 and test_perc > 0
    assert train_perc < 1 and val_perc < 1 and test_perc < 1
    train_mask = torch.full([size], fill_value=False)
    val_mask = torch.full([size], fill_value=False)
    test_mask = torch.full([size], fill_value=False)

    all_idxs = np.random.permutation(range(size))
    train_size = int(size * train_perc)
    val_size = int(size * val_perc)
    train_idxs, val_idxs, test_idxs = np.split(all_idxs, [train_size,
                                                          train_size + val_size])

    train_mask[train_idxs] = True
    val_mask[val_idxs] = True
    test_mask[test_idxs] = True

    assert train_mask.sum().item() + val_mask.sum().item() + test_mask.sum().item() == size
    return train_mask, val_mask, test_mask


def load_data(data_dir):
    data_dict = {'pos_edge_index': torch.load(data_dir / 'pos_edges_index.th'),
                 'neg_edge_index': torch.load(data_dir / 'neg_edges_index.th'),
                 'val_nex_edge_index': torch.load(data_dir / 'val_nex_edge_index.th'),
                 'test_nex_edge_index': torch.load(data_dir / 'test_nex_edge_index.th'),
                 'pos_train_mask': torch.load(data_dir / 'pos_train_mask.th'),
                 'pos_val_mask': torch.load(data_dir / 'pos_val_mask.th'),
                 'pos_test_mask': torch.load(data_dir / 'pos_test_mask.th'),
                 'neg_train_mask': torch.load(data_dir / 'neg_train_mask.th'),
                 'neg_val_mask': torch.load(data_dir / 'neg_val_mask.th'),
                 'neg_test_mask': torch.load(data_dir / 'neg_test_mask.th')}
    return data_dict


def setup_env(device_id, dataset_name, seed, num_splits, num_few_shot_train, num_few_shot_val, test_perc):
    device = torch.device("cuda" if torch.cuda.is_available() and device_id != "-1" else "cpu")
    # Creating folder to host run-specific files
    base_dir = pathlib.Path.cwd().parent
    my_run_id = uuid.uuid4()
    interim_data_dir = base_dir / 'data' / 'interim' / f"{my_run_id}"
    interim_data_dir.mkdir(exist_ok=True, parents=True)
    # Import dataset
    processed_data_dir = base_dir / 'data' / 'processed'
    data_dir = processed_data_dir / dataset_name
    data_dir = data_dir / f'seed_{seed}_num_splits_{num_splits}'
    data_dir = data_dir / f'train_{num_few_shot_train}_val_{num_few_shot_val}_test_{round(test_perc, 2)}'
    data_dir.mkdir(exist_ok=True, parents=True)
    return device, base_dir, interim_data_dir, data_dir


def move_data_to_device(data, device):
    for run_id in data:
        for split_type in ['train', 'val', 'test']:
            data[run_id][split_type]['edges'] = data[run_id][split_type]['edges'].to(device)
            data[run_id][split_type]['label'] = data[run_id][split_type]['label'].to(device)
        data[run_id]['train']['masked_edges'] = data[run_id]['train']['masked_edges'].to(
            device)
        data[run_id]['train']['masked_label'] = data[run_id]['train']['masked_label'].to(
            device)
    return data

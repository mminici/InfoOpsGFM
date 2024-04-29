import pathlib
import random
import uuid

import numpy as np
import torch

from node2vec import Node2Vec


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


def setup_env(device_id, dataset_name, seed, num_splits, is_few_shot, hyper_parameters):
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
    if is_few_shot:
        num_few_shot_train = hyper_parameters['num_few_shot_train']
        num_few_shot_val = hyper_parameters['num_few_shot_val']
        test_perc = hyper_parameters['test_perc']
        data_dir = data_dir / 'fsl' / f'train_{num_few_shot_train}_val_{num_few_shot_val}_test_{round(test_perc, 2)}'
        data_dir.mkdir(exist_ok=True, parents=True)
        return device, base_dir, interim_data_dir, data_dir
    train_perc = hyper_parameters['train_perc']
    val_perc = hyper_parameters['val_perc']
    test_perc = hyper_parameters['test_perc']
    data_dir = data_dir / f'train_{round(train_perc, 2)}_val_{round(val_perc, 2)}_test_{round(test_perc, 2)}'
    data_dir.mkdir(exist_ok=True, parents=True)
    return device, base_dir, interim_data_dir, data_dir


def move_data_to_device(data, device):
    return data


def load_node2vec_embeddings(data_dir, hyper_parameters):
    seed = hyper_parameters['seed']
    latent_dim = hyper_parameters['latent_dim']
    if (data_dir / f'node2vec_dim{latent_dim}_seed{seed}.npy').exists():
        print('Loading node2vec embed from disk...')
        return np.load(data_dir / f'node2vec_dim{latent_dim}_seed{seed}.npy')
    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(hyper_parameters['graph'], dimensions=hyper_parameters['latent_dim'],
                        walk_length=5, num_walks=10, workers=8, seed=seed)
    # Embed nodes
    model = node2vec.fit(window=8, min_count=1, batch_words=4, seed=seed)
    node_embeddings_node2vec = np.full(
        shape=(hyper_parameters['graph'].number_of_nodes(), hyper_parameters['latent_dim']),
        fill_value=None)
    for node_id in hyper_parameters['graph'].nodes():
        node_embeddings_node2vec[int(node_id)] = model.wv[node_id]
    np.save(data_dir / f'node2vec_dim{latent_dim}_seed{seed}.npy', node_embeddings_node2vec)
    return node_embeddings_node2vec

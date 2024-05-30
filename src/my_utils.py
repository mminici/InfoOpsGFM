import pathlib
import random
import shutil
import pickle
import uuid

import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import mlflow

from node2vec import Node2Vec
from torch_geometric.utils import from_networkx
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.transforms.add_positional_encoding import AddRandomWalkPE
from sklearn.decomposition import TruncatedSVD
from text_embed_util import get_tweet_embed


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


def setup_env(device_id, dataset_name, hyper_parameters):
    # seed, num_splits = hyper_parameters['seed'], hyper_parameters['num_splits']
    device = torch.device("cuda" if torch.cuda.is_available() and device_id != "-1" else "cpu")
    # Creating folder to host run-specific files
    base_dir = pathlib.Path.cwd().parent
    my_run_id = uuid.uuid4()
    interim_data_dir = base_dir / 'data' / 'interim' / f"{my_run_id}"
    interim_data_dir.mkdir(exist_ok=True, parents=True)
    # Import dataset
    processed_data_dir = base_dir / 'data' / 'processed'
    data_dir = processed_data_dir / dataset_name
    # data_dir = data_dir / f'seed_{seed}_num_splits_{num_splits}'
    # train_perc = hyper_parameters['train_perc']
    # val_perc = hyper_parameters['val_perc']
    # test_perc = hyper_parameters['test_perc']
    # data_dir = data_dir / f'train_{round(train_perc, 2)}_val_{round(val_perc, 2)}_test_{round(test_perc, 2)}'
    # data_dir.mkdir(exist_ok=True, parents=True)
    return device, base_dir, interim_data_dir, data_dir


def move_data_to_device(data, device):
    data['labels'] = torch.FloatTensor(data['labels']).to(device)
    return data


def load_node2vec_embeddings(data_dir, hyper_parameters):
    seed = hyper_parameters['seed']
    latent_dim = hyper_parameters['latent_dim']
    if (data_dir / f'node2vec_dim{latent_dim}_seed{seed}.npy').exists():
        print('Loading node2vec embed from disk...')
        return np.load(data_dir / f'node2vec_dim{latent_dim}_seed{seed}.npy', allow_pickle=True)
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


def get_edge_index_from_networkx(network):
    return from_networkx(network).edge_index


def create_spectral_features(
        graph, hidden_dim
) -> torch.FloatTensor:
    # Step 2. Compute Singular Value Decomposition of the adjacency matrix
    num_nodes = graph.number_of_nodes()
    adj_matrix = nx.to_numpy_array(graph)
    row, col = np.where(adj_matrix == 1.)
    sparse_adj_matrix = sp.coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(num_nodes, num_nodes))
    svd = TruncatedSVD(n_components=hidden_dim, n_iter=128)
    svd.fit(sparse_adj_matrix)
    node_features = svd.components_.T
    return torch.FloatTensor(node_features)


def degree_to_one_hot(degree_dict, num_buckets):
    max_node_id = max(degree_dict.keys()) + 1
    degrees_array = np.zeros((max_node_id, num_buckets), dtype=int)

    # Calculate percentiles
    values = np.array(list(degree_dict.values()))
    percentiles = np.percentile(values, np.linspace(0, 100, num_buckets))

    # Assign one-hot vectors for each node
    for node_id, degree in degree_dict.items():
        bucket = np.searchsorted(percentiles, degree, side="right") - 1
        degrees_array[node_id][bucket] = 1

    return torch.FloatTensor(degrees_array)


def get_gnn_embeddings(data_dir, hyper_parameters):
    embed_type = hyper_parameters['type']
    path_to_embed = data_dir / 'nodefeatures' / f'gnn{embed_type}.pth'
    if 'aggr_type' in hyper_parameters and hyper_parameters['aggr_type'] == 'max':
        path_to_embed = data_dir / 'nodefeatures' / f'gnn{embed_type}MAX.pth'
    if 'rw' in embed_type:
        latent_dim = hyper_parameters['latent_dim']
        path_to_embed = data_dir / 'nodefeatures' / f'gnn{embed_type}_rw{latent_dim}.pth'
    path_to_embed.parent.mkdir(parents=True, exist_ok=True)
    if path_to_embed.exists():
        print(f'Loading gnn({embed_type}) embed from disk...')
        return torch.load(path_to_embed)
    print(f'Compute gnn{embed_type} embed...')
    if embed_type == 'positional_onehot':
        num_nodes = hyper_parameters['num_nodes']
        node_features = torch.eye(num_nodes).float()
    elif embed_type == 'positional_random':
        num_nodes = hyper_parameters['num_nodes']
        latent_dim = hyper_parameters['latent_dim']
        node_features = torch.rand((num_nodes, latent_dim)).float()
    elif embed_type == 'positional_spectral':
        latent_dim = hyper_parameters['latent_dim']
        node_features = create_spectral_features(hyper_parameters['graph'], latent_dim)
    elif embed_type == 'positional_degree':
        latent_dim = hyper_parameters['latent_dim']
        node_features = degree_to_one_hot(dict(nx.degree(hyper_parameters['graph'])), latent_dim)
    elif embed_type == 'positional_rw':
        latent_dim = hyper_parameters['latent_dim']
        feature_generator = AddRandomWalkPE(latent_dim)
        graph_data = from_networkx(hyper_parameters['graph'])
        graph_data = feature_generator(graph_data)
        node_features = graph_data.random_walk_pe
    elif embed_type == 'tweets':
        node_features = get_tweet_embed(hyper_parameters['base_dir'], hyper_parameters['dataset_name'],
                                        hyper_parameters['noderemapping'], hyper_parameters['noderemapping_rev'],
                                        hyper_parameters['num_cores'], hyper_parameters['num_tweet_to_sample'],
                                        hyper_parameters['aggr_type'], hyper_parameters['device'])
    else:
        raise Exception(f'Embed type: {embed_type} not available yet.')
    torch.save(node_features, path_to_embed)
    return node_features


def get_edge_index(graph, data_dir, type=None):
    if type is None:
        fname = 'edge_index.th'
    else:
        fname = f'edge_index{type}.th'
    if not (data_dir / fname).exists():
        print(str(data_dir / fname) + ' does not exist. Computing it now...')
        edge_index = get_edge_index_from_networkx(graph)
        torch.save(edge_index, data_dir / fname)
        return edge_index
    else:
        print('Loading ' + str(data_dir / fname))
        return torch.load(data_dir / fname)


def _get_best_result(test_logger, metric_to_optimize):
    return test_logger.get_metric_stats(metric_to_optimize)[0]


def _save_best_result(path, test_logger, metric_to_optimize):
    with open(path, 'wb') as file:
        pickle.dump(test_logger.get_metric_stats(metric_to_optimize)[0], file)


def _load_best_result(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def _save_all_models(num_splits, interim_data_dir, best_model_datadir):
    for run_id in range(num_splits):
        best_model_path = interim_data_dir / f'model{run_id}.pth'
        shutil.copyfile(str(best_model_path), str(best_model_datadir / f'model{run_id}.pth'))


def update_best_model_snapshot(data_dir, metric_to_optimize, test_logger, num_splits, interim_data_dir):
    best_model_datadir = data_dir / f'best_models_{metric_to_optimize}'
    best_model_datadir.mkdir(parents=True, exist_ok=True)
    dir_is_empty = not any(best_model_datadir.iterdir())
    has_best_performance = (best_model_datadir / 'test_performance.pkl').exists()
    if (dir_is_empty or not has_best_performance) or _get_best_result(test_logger,
                                                                      metric_to_optimize) > _load_best_result(
            best_model_datadir / 'test_performance.pkl'):
        _save_all_models(num_splits, interim_data_dir, best_model_datadir)
        _save_best_result(best_model_datadir / 'test_performance.pkl', test_logger, metric_to_optimize)


def save_metrics(logger, interim_data_dir, split_type):
    print(f'{split_type} set: ')
    for metric_name in logger.test_metrics_dict:
        avg_val, std_val = logger.get_metric_stats(metric_name)
        mlflow.log_metric(metric_name + '_avg', avg_val)
        mlflow.log_metric(metric_name + '_std', std_val)
        np.save(file=interim_data_dir / f'val_{metric_name}' if split_type == 'VAL' else metric_name,
                arr=np.array(logger.test_metrics_dict[metric_name]))
        mlflow.log_artifact(
            interim_data_dir / f'val_{metric_name}.npy' if split_type == 'VAL' else f'{metric_name}.npy')
        print(f'[{split_type}] {metric_name}: {avg_val}+-{std_val}')


def create_data_loader_for_hgnn(datasets, graph_list, node_features, node_labels, data_dir, device, batch_size=None):
    data = HeteroData()
    data['node'].x = node_features
    data['node'].y = node_labels
    for graph_name in graph_list:
        data['node', graph_name, 'node'].edge_index = get_edge_index(datasets[graph_name], data_dir,
                                                                     type=graph_name).to(device).long()
    return data
    # return DataLoader([data], batch_size=batch_size, shuffle=True)


import os
import shutil
import mlflow
import networkx as nx
import numpy as np

from tqdm import tqdm
from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression
from my_utils import set_seed, setup_env
from data_loader import create_data_loader
from model_eval import get_best_threshold, eval_pred, TestLogMetrics


# noinspection PyShadowingNames
def run_experiment(dataset_name='UAE_sample',
                   seed=0,
                   train_few_shot_samples=10,
                   val_few_shot_samples=10,
                   test_perc=0.2,
                   hidden_dim=32,
                   num_splits=5,
                   metric_to_optimize='f1_macro',
                   device_id='1',
                   overwrite_data=False):
    # Save parameters
    mlflow.log_param('model', 'Node2Vec')
    mlflow.log_param('dataset_name', dataset_name)
    mlflow.log_param('overwrite_data', overwrite_data)
    mlflow.log_param('train_few_shot_samples', train_few_shot_samples)
    mlflow.log_param('val_few_shot_samples', val_few_shot_samples)
    mlflow.log_param('test_perc', test_perc)
    mlflow.log_param('hidden_dim', hidden_dim)
    mlflow.log_param('seed', seed)
    mlflow.log_param('num_splits', num_splits)
    mlflow.log_param('metric_to_optimize', metric_to_optimize)

    # set seed for reproducibility
    set_seed(seed)
    # set device
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    device, base_dir, interim_data_dir, data_dir = setup_env(device_id, dataset_name, seed, num_splits,
                                                             train_few_shot_samples, val_few_shot_samples,
                                                              test_perc)
    print(data_dir)
    # Create data loader for signed datasets
    datasets = create_data_loader(dataset_name, base_dir, data_dir,
                                  hyper_params={'overwrite_data': overwrite_data,
                                                'train_few_shot_samples': train_few_shot_samples,
                                                'val_few_shot_samples': val_few_shot_samples,
                                                'test_perc': test_perc,
                                                'seed': seed,
                                                'num_splits': num_splits,
                                                },
                                  device=device)
    # Remap nodeIDs
    node_ids_list = list(datasets['graph'].nodes())
    # Relabel each node from 0 to N-1
    node_remapping = {node_ids_list[i]: i for i in range(len(node_ids_list))}
    datasets['graph'] = nx.relabel_nodes(datasets['graph'], node_remapping)

    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(datasets['graph'], dimensions=hidden_dim, walk_length=5, num_walks=10, workers=8, seed=seed)
    # Embed nodes
    model = node2vec.fit(window=8, min_count=1, batch_words=4, seed=seed)
    node_embeddings_node2vec = np.full(shape=(datasets['graph'].number_of_nodes(), hidden_dim), fill_value=None)
    for node_id in datasets['graph'].nodes():
        node_embeddings_node2vec[int(node_id)] = model.wv[node_id]

    # Initialize metrics loggers
    test_logger = TestLogMetrics(num_splits, ['accuracy', 'precision', 'f1_macro', 'f1_micro'])
    for run_id in tqdm(range(num_splits), 'data split'):
        # Init binary classifier
        node_classifier = LogisticRegression()
        node_classifier.fit(node_embeddings_node2vec[datasets['splits'][run_id]['train']],
                            datasets['labels'][datasets['splits'][run_id]['train']])
        test_pred = node_classifier.predict(node_embeddings_node2vec)

        # Compute test statistics
        test_metrics = eval_pred(datasets['labels'], test_pred, datasets['splits'][run_id]['test'])
        for metric_name in test_metrics:
            test_logger.update(metric_name, run_id, test_metrics[metric_name])

    print('Test set: ')
    for metric_name in test_logger.test_metrics_dict:
        avg_val, std_val = test_logger.get_metric_stats(metric_name)
        mlflow.log_metric(metric_name + '_avg', avg_val)
        mlflow.log_metric(metric_name + '_std', std_val)
        np.save(file=interim_data_dir / f'{metric_name}', arr=np.array(test_logger.test_metrics_dict[metric_name]))
        mlflow.log_artifact(interim_data_dir / f'{metric_name}.npy')
        print(f'Test {metric_name}: {avg_val}+-{std_val}')


if __name__ == '__main__':
    # Run input parameters
    dataset_name = 'cuba'
    train_few_shot_samples = [3, 5, 10, 20]
    test_perc = 0.2
    # val_few_shot_samples = [5, 10, 25, 50, 100, 250, 500, 1000]
    seed = [0, ]
    num_splits = [20, ]
    device_id = '1'
    metric_to_optimize = 'f1_macro'
    for seed_val in seed:
        mlflow.set_experiment(f'{dataset_name}-Node2Vec-{seed_val}')
        for train_few_shot_samples_val in train_few_shot_samples:
            val_few_shot_samples_val = train_few_shot_samples_val
            for num_splits_val in num_splits:
                with mlflow.start_run():
                    exp_dir = run_experiment(dataset_name=dataset_name,
                                             seed=seed_val,
                                             train_few_shot_samples=train_few_shot_samples_val,
                                             val_few_shot_samples=val_few_shot_samples_val,
                                             test_perc=test_perc,
                                             num_splits=num_splits_val,
                                             metric_to_optimize='f1_macro',
                                             device_id='1',
                                             overwrite_data=False
                                             )
                    try:
                        shutil.rmtree(exp_dir, ignore_errors=True)
                    except OSError as e:
                        print("Error: %s - %s." % (e.filename, e.strerror))

import os
import shutil
import mlflow
import networkx as nx
import numpy as np

from tqdm import tqdm
from my_utils import set_seed, setup_env
from data_loader import create_data_loader
from model_eval import get_best_threshold, eval_pred, TestLogMetrics


# noinspection PyShadowingNames
def run_experiment(dataset_name='UAE_sample',
                   seed=0,
                   train_few_shot_samples=10,
                   val_few_shot_samples=10,
                   test_perc=0.2,
                   num_splits=5,
                   metric_to_optimize='f1_macro',
                   device_id='1',
                   overwrite_data=False):
    # Save parameters
    mlflow.log_param('model', 'NodePruning')
    mlflow.log_param('dataset_name', dataset_name)
    mlflow.log_param('overwrite_data', overwrite_data)
    mlflow.log_param('train_few_shot_samples', train_few_shot_samples)
    mlflow.log_param('val_few_shot_samples', val_few_shot_samples)
    mlflow.log_param('test_perc', test_perc)
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
    # Create data loader for datasets
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

    # Compute node centrality values
    # TODO: take the type of centrality as input parameter
    centrality_values = nx.eigenvector_centrality(datasets['graph'])
    # Transform into a list
    centrality_val_list = [-1] * datasets['graph'].number_of_nodes()
    for node_id in tqdm(centrality_values):
        centrality_val_list[node_id] = centrality_values[node_id]
    centrality_val_list = np.array(centrality_val_list)

    # Perform prediction based on various centrality threshold
    predicted_labels_list = []
    for percentile in tqdm(np.arange(0, 100, 0.5)):
        centrality_threshold = np.percentile(centrality_val_list, percentile)
        predicted_labels = np.full(shape=datasets['graph'].number_of_nodes(), fill_value=1)
        coordinated_users = np.where(centrality_val_list <= centrality_threshold)[0]
        predicted_labels[coordinated_users] = 0
        predicted_labels_list.append(np.copy(predicted_labels))

    # Initialize metrics loggers
    val_logger = TestLogMetrics(num_splits, ['accuracy', 'precision', 'f1_macro', 'f1_micro'])
    test_logger = TestLogMetrics(num_splits, ['accuracy', 'precision', 'f1_macro', 'f1_micro'])

    for run_id in tqdm(range(num_splits), 'data split'):
        # Since this is an unsupervised baseline, we merge training and validation
        unsupervised_mask = np.logical_or(datasets['splits'][run_id]['train'], datasets['splits'][run_id]['val'])
        # Select the best threshold according to the eval on train+val sets
        best_val_threshold = get_best_threshold(datasets['labels'],
                                                predicted_labels_list, unsupervised_mask, metric_to_optimize)
        val_metrics = eval_pred(datasets['labels'], predicted_labels_list[best_val_threshold], unsupervised_mask)
        # Compute test statistics
        test_metrics = eval_pred(datasets['labels'], predicted_labels_list[best_val_threshold],
                                 datasets['splits'][run_id]['test'])
        for metric_name in test_metrics:
            test_logger.update(metric_name, run_id, test_metrics[metric_name])
            val_logger.update(metric_name, run_id, val_metrics[metric_name])

    # Simulation ended, report metrics on test set for the best model
    print('Val set: ')
    for metric_name in val_logger.test_metrics_dict:
        avg_val, std_val = val_logger.get_metric_stats(metric_name)
        mlflow.log_metric('val_' + metric_name + '_avg', avg_val)
        mlflow.log_metric('val_' + metric_name + '_std', std_val)
        np.save(file=interim_data_dir / f'val_{metric_name}', arr=np.array(val_logger.test_metrics_dict[metric_name]))
        mlflow.log_artifact(interim_data_dir / f'val_{metric_name}.npy')
        print(f'Val {metric_name}: {avg_val}+-{std_val}')

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
        mlflow.set_experiment(f'{dataset_name}-NodePruning-{seed_val}')
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
                                             metric_to_optimize=metric_to_optimize,
                                             device_id=device_id,
                                             overwrite_data=False
                                             )
                    try:
                        shutil.rmtree(exp_dir, ignore_errors=True)
                    except OSError as e:
                        print("Error: %s - %s." % (e.filename, e.strerror))

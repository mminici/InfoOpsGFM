import os
import mlflow
import shutil
import numpy as np
import networkx as nx

from data_loader import create_data_loader
from model_eval import TestLogMetrics, eval_pred, get_best_threshold
from my_utils import set_seed, setup_env, save_metrics
from tqdm import tqdm

DEFAULT_HYPERPARAMETERS = {'train_perc': 0.7,
                           'val_perc': 0.15,
                           'test_perc': 0.15,
                           'overwrite_data': False}
DEFAULT_TRAIN_HYPERPARAMETERS = {'metric_to_optimize': 'f1_macro'}
DEFAULT_MODEL_HYPERPARAMETERS = {}


# noinspection PyShadowingNames
def run_experiment(dataset_name='cuba',
                   is_few_shot=False,
                   num_splits=10,
                   device_id="",
                   seed=0,
                   hyper_parameters=None,
                   train_hyperparameters=None
                   ):
    # Start experiment
    if train_hyperparameters is None:
        train_hyperparameters = DEFAULT_TRAIN_HYPERPARAMETERS
    if hyper_parameters is None:
        hyper_parameters = DEFAULT_HYPERPARAMETERS
    # save parameters
    mlflow.log_param('dataset_name', dataset_name)

    # set seed for reproducibility
    set_seed(seed)
    # set device
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    device, base_dir, interim_data_dir, data_dir = setup_env(device_id, dataset_name, seed, num_splits,
                                                             is_few_shot,
                                                             hyper_parameters)
    print(data_dir)
    # Create data loader for signed datasets
    datasets = create_data_loader(dataset_name, base_dir, data_dir, hyper_parameters)

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

    # Create loggers
    val_logger = TestLogMetrics(num_splits, ['accuracy', 'precision', 'f1_macro', 'f1_micro'])
    test_logger = TestLogMetrics(num_splits, ['accuracy', 'precision', 'f1_macro', 'f1_micro'])

    for run_id in range(num_splits):
        print(f'Split {run_id + 1}/{num_splits}')

        # Since this is an unsupervised baseline, we merge training and validation
        unsupervised_mask = np.logical_or(datasets['splits'][run_id]['train'], datasets['splits'][run_id]['val'])
        # Select the best threshold according to the eval on train+val sets
        best_val_threshold = get_best_threshold(datasets['labels'],
                                                predicted_labels_list,
                                                unsupervised_mask,
                                                train_hyperparameters["metric_to_optimize"])
        val_metrics = eval_pred(datasets['labels'], predicted_labels_list[best_val_threshold], unsupervised_mask)
        # Compute test statistics
        test_metrics = eval_pred(datasets['labels'], predicted_labels_list[best_val_threshold],
                                 datasets['splits'][run_id]['test'])
        for metric_name in test_metrics:
            test_logger.update(metric_name, run_id, test_metrics[metric_name])
            val_logger.update(metric_name, run_id, val_metrics[metric_name])

    # Save metrics
    save_metrics(val_logger, interim_data_dir, 'VAL')
    save_metrics(test_logger, interim_data_dir, 'TEST')


if __name__ == '__main__':
    # Run input parameters
    dataset_name = 'cuba'
    train_perc = 0.70
    val_perc = 0.15
    test_perc = 0.15
    overwrite_data = False
    is_few_shot = False
    seed = [0, ]
    num_splits = [10, ]
    # General hyperparameters
    hyper_parameters = {'train_perc': train_perc, 'val_perc': val_perc, 'test_perc': test_perc,
                        'overwrite_data': overwrite_data, 'traces_list': ['coRT'],
                        'extract_largest_connected_component': True,
                        'is_few_shot': is_few_shot}
    # optimization hyperparameters
    train_hyper_parameters = {'metric_to_optimize': 'f1_macro'}
    # model hyperparameters
    model_hyper_parameters = {}
    for seed_val in seed:
        mlflow.set_experiment(f'{dataset_name}-NodePruning-{seed_val}')
        for num_splits_val in num_splits:
            hyper_parameters['num_splits'] = num_splits_val
            with mlflow.start_run():
                exp_dir = run_experiment(dataset_name=dataset_name,
                                         is_few_shot=is_few_shot,
                                         num_splits=num_splits_val,
                                         seed=seed_val,
                                         hyper_parameters=hyper_parameters,
                                         train_hyperparameters=train_hyper_parameters
                                         )
                try:
                    shutil.rmtree(exp_dir, ignore_errors=True)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))

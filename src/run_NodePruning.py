import os
import argparse
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
def main(dataset_name='cuba',
         num_splits=10,
         device_id="",
         seed=0,
         hyper_params=None,
         train_hyperparams=None
         ):
    # Start experiment
    if train_hyperparams is None:
        train_hyperparams = DEFAULT_TRAIN_HYPERPARAMETERS
    if hyper_params is None:
        hyper_params = DEFAULT_HYPERPARAMETERS
    # save parameters
    mlflow.log_param('dataset_name', dataset_name)
    # set seed for reproducibility
    set_seed(seed)
    # set device
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    _, base_dir, interim_data_dir, data_dir = setup_env('', dataset_name, hyper_params)
    print(data_dir)
    # Create data loader for signed datasets
    datasets = create_data_loader(data_dir, hyper_params['tsim_th'])
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
                                                train_hyperparams["metric_to_optimize"])
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
    parser = argparse.ArgumentParser(description="Preprocess dataset to produce train-val-test split")
    parser.add_argument('-dataset_name', '--dataset', type=str, help='Dataset', default='cuba')
    parser.add_argument('-seed', '--seed', type=int, help='Random seed', default=12121995)
    parser.add_argument('-train_perc', '--train', type=float, help='Training percentage', default=.6)
    parser.add_argument('-val_perc', '--val', type=float, help='Validation percentage', default=.2)
    parser.add_argument('-test_perc', '--test', type=float, help='Test percentage', default=.2)
    parser.add_argument('-num_splits', '--splits', type=int, help='Num of train-val-test splits', default=5)
    parser.add_argument('-tweet_sim_threshold', '--tsim_th', type=float, help='Threshold over which we retain an edge '
                                                                              'in tweet similarity network',
                        default=.99)
    parser.add_argument('-metric_to_optimize', '--val_metric', type=str, help='Metric to optimize', default='f1_macro')
    args = parser.parse_args()
    # Run input parameters
    # General hyperparameters
    hyper_parameters = {'train_perc': args.train, 'val_perc': args.val, 'test_perc': args.test,
                        'num_splits': args.splits, 'tsim_th': args.tsim_th}
    # optimization hyperparameters
    train_hyper_parameters = {'metric_to_optimize': 'f1_macro'}
    # model hyperparameters
    model_hyper_parameters = {}
    mlflow.set_experiment(f'{args.dataset}-NodePruning-{args.seed}')
    with mlflow.start_run():
        exp_dir = main(dataset_name=args.dataset,
                       num_splits=args.splits,
                       device_id="",
                       seed=args.seed,
                       hyper_params=hyper_parameters,
                       train_hyperparams=train_hyper_parameters
                       )
        try:
            shutil.rmtree(exp_dir, ignore_errors=True)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

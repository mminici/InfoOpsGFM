import os
import mlflow
import shutil
import numpy as np

from data_loader import create_data_loader
from model_eval import TestLogMetrics, eval_pred
from my_utils import set_seed, setup_env, load_node2vec_embeddings, save_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

DEFAULT_HYPERPARAMETERS = {'train_perc': 0.7,
                           'val_perc': 0.15,
                           'test_perc': 0.15,
                           'overwrite_data': False,
                           'extract_largest_connected_component': True,
                           'is_few_shot': False,
                           'num_splits': 20}
DEFAULT_TRAIN_HYPERPARAMETERS = {}
DEFAULT_MODEL_HYPERPARAMETERS = {'latent_dim': 32}


def create_model(model_hyperparameters):
    if model_hyperparameters["model_name"] == 'LR':
        return LogisticRegression()
    elif model_hyperparameters["model_name"] == 'RF':
        return RandomForestClassifier()
    model_name = model_hyperparameters["model_name"]
    raise Exception(f'{model_name} not allowed.')


# noinspection PyShadowingNames
def run_experiment(dataset_name='cuba',
                   is_few_shot=False,
                   num_splits=10,
                   device_id="",
                   seed=0,
                   hyper_parameters=None,
                   train_hyperparameters=None,
                   model_hyper_parameters=None
                   ):
    # Start experiment
    if model_hyper_parameters is None:
        model_hyper_parameters = DEFAULT_MODEL_HYPERPARAMETERS
    if train_hyperparameters is None:
        train_hyperparameters = DEFAULT_TRAIN_HYPERPARAMETERS
    if hyper_parameters is None:
        hyper_parameters = DEFAULT_HYPERPARAMETERS
    # save parameters
    mlflow.log_param('dataset_name', dataset_name)
    mlflow.log_param('latent_dim', model_hyper_parameters['latent_dim'])

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

    # Get Node2Vec embeddings
    node_embeddings_node2vec = load_node2vec_embeddings(data_dir, {'graph': datasets['graph'],
                                                                   'latent_dim': model_hyper_parameters['latent_dim'],
                                                                   'seed': seed})

    # Create loggers
    val_logger = TestLogMetrics(num_splits, ['accuracy', 'precision', 'f1_macro', 'f1_micro'])
    test_logger = TestLogMetrics(num_splits, ['accuracy', 'precision', 'f1_macro', 'f1_micro'])

    for run_id in range(num_splits):
        print(f'Split {run_id + 1}/{num_splits}')

        # Create the model
        model = create_model(model_hyper_parameters)
        model.fit(node_embeddings_node2vec[datasets['splits'][run_id]['train']],
                  datasets['labels'][datasets['splits'][run_id]['train']])
        pred = model.predict(node_embeddings_node2vec)
        # Evaluate perfomance on val set
        val_metrics = eval_pred(datasets['labels'], pred, datasets['splits'][run_id]['val'])
        for metric_name in val_metrics:
            val_logger.update(metric_name, run_id, val_metrics[metric_name])
        # Evaluate perfomance on test set
        test_metrics = eval_pred(datasets['labels'], pred, datasets['splits'][run_id]['test'])
        for metric_name in test_metrics:
            test_logger.update(metric_name, run_id, test_metrics[metric_name])

    # Save metrics
    save_metrics(val_logger, interim_data_dir, 'VAL')
    save_metrics(test_logger, interim_data_dir, 'TEST')


if __name__ == '__main__':
    # Run input parameters
    dataset_name = 'UAE_sample'
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
    train_hyper_parameters = {}
    # model hyperparameters
    latent_dim = 64
    model_hyper_parameters = {'latent_dim': latent_dim, 'model_name': 'LR'}
    for seed_val in seed:
        mlflow.set_experiment(f'{dataset_name}-Node2Vec-{seed_val}')
        for num_splits_val in num_splits:
            hyper_parameters['num_splits'] = num_splits_val
            with mlflow.start_run():
                exp_dir = run_experiment(dataset_name=dataset_name,
                                         is_few_shot=is_few_shot,
                                         num_splits=num_splits_val,
                                         seed=seed_val,
                                         hyper_parameters=hyper_parameters,
                                         train_hyperparameters=train_hyper_parameters,
                                         model_hyper_parameters=model_hyper_parameters
                                         )
                try:
                    shutil.rmtree(exp_dir, ignore_errors=True)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))

import os
import argparse
import mlflow
import shutil

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
def main(dataset_name='cuba',
         num_splits=10,
         seed=0,
         hyper_params=None,
         train_hyperparams=None,
         model_hyperparams=None
         ):
    # Start experiment
    if model_hyperparams is None:
        model_hyperparams = DEFAULT_MODEL_HYPERPARAMETERS
    if train_hyperparams is None:
        train_hyperparams = DEFAULT_TRAIN_HYPERPARAMETERS
    if hyper_params is None:
        hyper_params = DEFAULT_HYPERPARAMETERS
    # save parameters
    mlflow.log_param('dataset_name', dataset_name)
    mlflow.log_param('latent_dim', model_hyperparams['latent_dim'])

    # set seed for reproducibility
    set_seed(seed)
    # set device
    _, base_dir, interim_data_dir, data_dir = setup_env('', dataset_name, hyper_params)
    print(data_dir)
    # Create data loader for signed datasets
    datasets = create_data_loader(data_dir, hyper_params['tsim_th'])
    print('Computing node2vec embedding...')
    # Get Node2Vec embeddings
    node_embeddings_node2vec = load_node2vec_embeddings(data_dir, {'graph': datasets['graph'],
                                                                   'latent_dim': model_hyperparams['latent_dim'],
                                                                   'seed': seed})
    # Create loggers
    val_logger = TestLogMetrics(num_splits, ['accuracy', 'precision', 'f1_macro', 'f1_micro'])
    test_logger = TestLogMetrics(num_splits, ['accuracy', 'precision', 'f1_macro', 'f1_micro'])

    for run_id in range(num_splits):
        print(f'Split {run_id + 1}/{num_splits}')
        # Creating the model
        model = create_model(model_hyperparams)
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
    parser = argparse.ArgumentParser(description="Preprocess dataset to produce train-val-test split")
    parser.add_argument('-dataset_name', '--dataset', type=str, help='Dataset', default='cuba')
    parser.add_argument('-seed', '--seed', type=int, help='Random seed', default=12121995)
    parser.add_argument('-train_perc', '--train', type=float, help='Training percentage', default=.6)
    parser.add_argument('-val_perc', '--val', type=float, help='Validation percentage', default=.2)
    parser.add_argument('-test_perc', '--test', type=float, help='Test percentage', default=.2)
    parser.add_argument('-num_splits', '--splits', type=int, help='Num of train-val-test splits', default=5)
    parser.add_argument('-metric_to_optimize', '--val_metric', type=str, help='Metric to optimize', default='f1_macro')
    parser.add_argument('-latent_dim', '--latent', type=int, help='Latent dimension', default=100)
    parser.add_argument('-tweet_sim_threshold', '--tsim_th', type=float, help='Threshold over which we retain an edge '
                                                                              'in tweet similarity network',
                        default=.99)
    args = parser.parse_args()
    # Run input parameters
    # General hyperparameters
    hyper_parameters = {'train_perc': args.train, 'val_perc': args.val, 'test_perc': args.test,
                        'tsim_th': args.tsim_th, 'val_metric': args.val_metric}
    # optimization hyperparameters
    train_hyper_parameters = {}
    # model hyperparameters
    model_hyper_parameters = {'latent_dim': args.latent, 'model_name': 'RF'}
    mlflow.set_experiment(f'{args.dataset}-Node2Vec-{args.seed}')
    hyper_parameters['num_splits'] = args.splits
    with mlflow.start_run():
        exp_dir = main(dataset_name=args.dataset,
                       num_splits=args.splits,
                       seed=args.seed,
                       hyper_params=hyper_parameters,
                       train_hyperparams=train_hyper_parameters,
                       model_hyperparams=model_hyper_parameters
                       )
        try:
            shutil.rmtree(exp_dir, ignore_errors=True)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

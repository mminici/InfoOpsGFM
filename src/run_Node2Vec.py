import argparse
import mlflow
import shutil
import os
import torch
import numpy as np

from data_loader import create_data_loader
from model_eval import TestLogMetrics, eval_pred, TrainLogMetrics
from my_utils import set_seed, setup_env, save_metrics, handle_isolated_nodes, get_edge_index, move_data_to_device
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.nn import Node2Vec

DEFAULT_HYPERPARAMETERS = {'train_perc': 0.6,
                           'val_perc': 0.2,
                           'test_perc': 0.2,
                           'num_splits': 5}
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
         model_hyperparams=None,
         device_id='3'
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
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    device, base_dir, interim_data_dir, data_dir = setup_env(device_id, dataset_name, hyper_params)
    print(data_dir)
    # Create data loader for signed datasets
    datasets = create_data_loader(data_dir, hyper_params['tsim_th'],
                                  hyper_params['train_perc'], hyper_params['undersampling'])
    # Transfer data to device
    datasets = move_data_to_device(datasets, device)
    print('Computing node2vec embedding...')
    # Preprocessing: all isolated nodes must be rewired
    _, network = handle_isolated_nodes(datasets['graph'])
    # Get edge index representation
    print('Get edge index from graph ({}N {}E)'.format(network.number_of_nodes(),
                                                       network.number_of_edges()))
    edge_index = get_edge_index(network, data_dir)
    edge_index = edge_index.to(device)
    # Create numpy version of labels for the validation phase
    numpy_labels = datasets['labels'].long().detach().cpu().numpy()
    # Create loggers
    train_logger = TrainLogMetrics(hyper_params['num_splits'], ['supervised'])
    val_logger = TestLogMetrics(num_splits, ['accuracy', 'precision', 'f1_macro', 'f1_micro'])
    test_logger = TestLogMetrics(num_splits, ['accuracy', 'precision', 'f1_macro', 'f1_micro'])
    # get training hyperparameters
    num_epochs = train_hyperparams['num_epochs']
    metric_to_optimize = train_hyperparams['metric_to_optimize']
    for run_id in range(num_splits):
        print(f'Split {run_id + 1}/{num_splits}')
        BEST_VAL_METRIC = -np.inf
        best_model_path = interim_data_dir / f'model{run_id}.pth'
        early_stopping_cnt = 0
        # Creating the model
        model = Node2Vec(
            edge_index,
            embedding_dim=128,
            walk_length=5,
            context_size=4,
            walks_per_node=10,
            num_negative_samples=1,
            p=1.0,
            q=1.0,
            sparse=True,
        ).to(device)
        num_workers = 4
        loader = model.loader(batch_size=128, shuffle=True, num_workers=num_workers)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=train_hyperparams['learning_rate'])
        for epoch in range(num_epochs):
            if early_stopping_cnt > train_hyperparams["early_stopping_limit"]:
                break
            model.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            train_logger.train_update(run_id, 'supervised', total_loss / len(loader))
            if epoch % train_hyperparams["check_loss_freq"] == 0:
                # Validation step
                model.eval()
                with torch.no_grad():
                    z = model().detach().cpu().numpy()
                    clf = create_model(model_hyperparams)
                    clf.fit(z[datasets['splits'][run_id]['train']],
                            numpy_labels[datasets['splits'][run_id]['train']])
                    pred = clf.predict(z)
                    val_metrics = eval_pred(numpy_labels, pred, datasets['splits'][run_id]['val'])
                    train_logger.val_update(run_id, val_metrics[train_hyperparams["metric_to_optimize"]])
                    if val_metrics[train_hyperparams["metric_to_optimize"]] > BEST_VAL_METRIC:
                        BEST_VAL_METRIC = val_metrics[train_hyperparams["metric_to_optimize"]]
                        torch.save(model.state_dict(), best_model_path)
                        early_stopping_cnt = 0
                    else:
                        early_stopping_cnt += 1
                    print(
                        f'Epoch {epoch}/{num_epochs} train_loss: {total_loss / len(loader)} -- val_{metric_to_optimize}: {val_metrics[metric_to_optimize]}')
            else:
                train_logger.val_update(run_id, 0.0)
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        with torch.no_grad():
            z = model().detach().cpu().numpy()
            clf = create_model(model_hyperparams)
            clf.fit(z[datasets['splits'][run_id]['train']],
                    numpy_labels[datasets['splits'][run_id]['train']])
            pred = clf.predict(z)
        # Evaluate perfomance on val set
        val_metrics = eval_pred(numpy_labels, pred, datasets['splits'][run_id]['val'])
        for metric_name in val_metrics:
            val_logger.update(metric_name, run_id, val_metrics[metric_name])
        # Evaluate perfomance on test set
        test_metrics = eval_pred(numpy_labels, pred, datasets['splits'][run_id]['test'])
        for metric_name in test_metrics:
            test_logger.update(metric_name, run_id, test_metrics[metric_name])

    # Save metrics
    save_metrics(val_logger, interim_data_dir, 'VAL')
    save_metrics(test_logger, interim_data_dir, 'TEST')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess dataset to produce train-val-test split")
    parser.add_argument('-device_id', '--device', type=str, help='GPU ID#', default='3')
    parser.add_argument('-dataset_name', '--dataset', type=str, help='Dataset', default='cuba')
    parser.add_argument('-seed', '--seed', type=int, help='Random seed', default=12121995)
    parser.add_argument('-train_perc', '--train', type=float, help='Training percentage', default=.6)
    parser.add_argument('-val_perc', '--val', type=float, help='Validation percentage', default=.2)
    parser.add_argument('-test_perc', '--test', type=float, help='Test percentage', default=.2)
    parser.add_argument('-num_splits', '--splits', type=int, help='Num of train-val-test splits', default=5)
    parser.add_argument('-latent_dim', '--latent', type=int, help='Latent dimension', default=128)
    parser.add_argument('-tweet_sim_threshold', '--tsim_th', type=float, help='Threshold over which we retain an edge '
                                                                              'in tweet similarity network',
                        default=.7)
    parser.add_argument('-num_epochs', '--epochs', type=int, help='#Training Epochs', default=1000)
    parser.add_argument('-learning_rate', '--lr', type=float, help='Optimizer Learning Rate', default=1e-2)
    parser.add_argument('-early_stopping_limit', '--early', type=int, help='Num patience steps', default=20)
    parser.add_argument('-check_loss_freq', '--check', type=int, help='Frequency validation check', default=1)
    parser.add_argument('-metric_to_optimize', '--val_metric', type=str, help='Metric to optimize', default='f1_macro')
    parser.add_argument('-under_sampling', '--under', help='undersampling percentage', default=None)
    args = parser.parse_args()
    # Run input parameters
    # General hyperparameters
    hyper_parameters = {'train_perc': args.train, 'val_perc': args.val, 'test_perc': args.test,
                        'tsim_th': args.tsim_th, 'val_metric': args.val_metric,
                        'undersampling': float(args.under) if args.under is not None else None}
    # optimization hyperparameters
    train_hyper_parameters = {'num_epochs': args.epochs, 'learning_rate': args.lr,
                              'early_stopping_limit': args.early, 'check_loss_freq': args.check,
                              'metric_to_optimize': args.val_metric}
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
                       model_hyperparams=model_hyper_parameters,
                       device_id=args.device
                       )
        try:
            shutil.rmtree(exp_dir, ignore_errors=True)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

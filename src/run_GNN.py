import os
import mlflow
import shutil
import torch
import numpy as np

from data_loader import create_data_loader
from models import GNN
from model_eval import TrainLogMetrics, TestLogMetrics, eval_pred
from my_utils import set_seed, setup_env, move_data_to_device, get_edge_index_from_networkx, get_gnn_embeddings, update_best_model_snapshot, save_metrics
from plot_utils import plot_losses

DEFAULT_HYPERPARAMETERS = {'train_perc': 0.7,
                           'val_perc': 0.15,
                           'test_perc': 0.15,
                           'overwrite_data': False,
                           'extract_largest_connected_component': True,
                           'is_few_shot': False,
                           'num_splits': 20}
DEFAULT_TRAIN_HYPERPARAMETERS = {'input_embed': 'positional', 'epochs': 1000, 'learning_rate': 1e-3,
                                 'early_stopping_limit': 10, 'check_loss_freq': 5}
DEFAULT_MODEL_HYPERPARAMETERS = {'gnn_type': 'gcn', 'latent_dim': 32, 'dropout': 0.2}


def create_model(model_hyperparameters):
    return GNN(num_node_features=model_hyperparameters['feature_dim'], hidden_dim=model_hyperparameters['latent_dim'],
               num_classes=2, dropout_p=model_hyperparameters['dropout'], gnn_type=model_hyperparameters['gnn_type'])


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
    mlflow.log_param('input_embed', train_hyperparameters['input_embed'])
    mlflow.log_param('gnn_type', model_hyper_parameters['gnn_type'])
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

    # Transfer data to device
    datasets = move_data_to_device(datasets, device)

    # Get edge index representation
    edge_index = get_edge_index_from_networkx(datasets['graph'])
    edge_index.to(device)

    # Get node features
    node_features = get_gnn_embeddings(data_dir, {'type': train_hyperparameters['input_embed'],
                                                  'latent_dim': model_hyper_parameters['latent_dim'],
                                                  'seed': seed, 'num_nodes': datasets['graph'].number_of_nodes(),
                                                  'graph': datasets['graph'],
                                                  'overwrite_data': hyper_parameters['overwrite_data']})
    node_features.to(device)
    model_hyper_parameters['feature_dim'] = node_features.shape[1]

    # Create loggers
    train_logger = TrainLogMetrics(num_splits, ['supervised'])
    val_logger = TestLogMetrics(num_splits, ['accuracy', 'precision', 'f1_macro', 'f1_micro'])
    test_logger = TestLogMetrics(num_splits, ['accuracy', 'precision', 'f1_macro', 'f1_micro'])
    # Create numpy version of labels for the validation phase
    numpy_labels = datasets['labels'].long().detach().cpu().numpy()

    # get training hyperparameters
    num_epochs = train_hyperparameters['num_epochs']
    metric_to_optimize = train_hyperparameters['metric_to_optimize']
    for run_id in range(num_splits):
        print(f'Split {run_id + 1}/{num_splits}')

        BEST_VAL_METRIC = -np.inf
        best_model_path = interim_data_dir / f'model{run_id}.pth'

        early_stopping_cnt = 0

        # Create the model
        model = create_model(model_hyper_parameters)
        model.to(device)
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=train_hyperparameters['learning_rate'])
        loss_fn = torch.nn.BCELoss()
        for epoch in range(num_epochs):
            if early_stopping_cnt > train_hyperparameters["early_stopping_limit"]:
                break
            model.train()
            optimizer.zero_grad()
            pred = model(node_features, edge_index).flatten()
            loss = loss_fn(pred[datasets['splits'][run_id]['train']],
                           datasets['labels'][datasets['splits'][run_id]['train']])
            loss.backward()
            optimizer.step()
            train_logger.train_update(run_id, 'supervised', loss.item())
            if epoch % train_hyperparameters["check_loss_freq"] == 0:
                # Validation step
                model.eval()
                with torch.no_grad():
                    pred = model(node_features, edge_index).detach().cpu().numpy().flatten()
                    val_metrics = eval_pred(numpy_labels, pred > 0.5, datasets['splits'][run_id]['val'])
                    train_logger.val_update(run_id, val_metrics[train_hyperparameters["metric_to_optimize"]])
                    if val_metrics[train_hyperparameters["metric_to_optimize"]] > BEST_VAL_METRIC:
                        BEST_VAL_METRIC = val_metrics[train_hyperparameters["metric_to_optimize"]]
                        torch.save(model.state_dict(), best_model_path)
                        early_stopping_cnt = 0
                    else:
                        early_stopping_cnt += 1
                    print(
                        f'Epoch {epoch}/{num_epochs} train_loss: {loss.item()} -- val_{metric_to_optimize}: {val_metrics[metric_to_optimize]}')
            else:
                train_logger.val_update(run_id, 0.0)
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        with torch.no_grad():
            pred = model(node_features, edge_index).detach().cpu().numpy().flatten()
        # Evaluate perfomance on val set
        val_metrics = eval_pred(numpy_labels, pred > 0.5, datasets['splits'][run_id]['val'])
        for metric_name in val_metrics:
            val_logger.update(metric_name, run_id, val_metrics[metric_name])
        # Evaluate perfomance on test set
        test_metrics = eval_pred(numpy_labels, pred > 0.5, datasets['splits'][run_id]['test'])
        for metric_name in test_metrics:
            test_logger.update(metric_name, run_id, test_metrics[metric_name])

    for split_num in range(num_splits):
        mlflow.log_artifact(interim_data_dir / f'model{split_num}.pth')  # store best model
        fig = plot_losses(
            train_values=[train_logger.train_loss_dict[split_num]['supervised']],
            val_values=[train_logger.val_metrics_dict[split_num]],
            train_labels=['supervised loss'],
            val_labels=[f'val {metric_to_optimize}'])
        fig.savefig(interim_data_dir / f'train_and_val_loss_curves{split_num}.png', dpi=800)
        fig.savefig(interim_data_dir / f'train_and_val_loss_curves{split_num}.pdf')
        mlflow.log_artifact(interim_data_dir / f'train_and_val_loss_curves{split_num}.png')
        mlflow.log_artifact(interim_data_dir / f'train_and_val_loss_curves{split_num}.pdf')

    # Save metrics
    save_metrics(val_logger, interim_data_dir, 'VAL')
    save_metrics(test_logger, interim_data_dir, 'TEST')

    # Save best models
    update_best_model_snapshot(data_dir, metric_to_optimize, test_logger, num_splits, interim_data_dir)


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
    train_hyper_parameters = {'input_embed': 'positional_degree', 'num_epochs': 1000, 'learning_rate': 1e-3,
                              'early_stopping_limit': 10, 'check_loss_freq': 5, 'metric_to_optimize': 'f1_macro'}
    # model hyperparameters
    latent_dim = 100
    gnn_type = 'gcn'
    model_hyper_parameters = {'gnn_type': gnn_type, 'latent_dim': latent_dim, 'dropout': 0.2}
    for seed_val in seed:
        mlflow.set_experiment(f'{dataset_name}-GNN-{seed_val}')
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

import os
import mlflow
import torch
import shutil
import numpy as np

from data_loader import create_data_loader
from model_eval import TrainLogMetrics, TestLogMetrics
from my_utils import set_seed, setup_env, move_data_to_device
from plot_utils import plot_losses


DEFAULT_HYPERPARAMETERS = {'train_perc': 0.7,
                           'val_perc': 0.15,
                           'test_perc': 0.15,
                           'overwrite_data': False}
DEFAULT_TRAIN_HYPERPARAMETERS = {'num_epochs': 100,
                                 'learning_rate': 1e-3,
                                 'early_stopping_limit': 10,
                                 'metric_to_optimize': 'auc_score',
                                  }
DEFAULT_MODEL_HYPERPARAMETERS = {'latent_dim': 32}

# noinspection PyShadowingNames
def run_experiment(dataset_name='cuba',
                   is_few_shot=False,
                   num_splits=10,
                   device_id="",
                   seed=0,
                   hyper_parameters=DEFAULT_HYPERPARAMETERS,
                   train_hyperparameters=DEFAULT_TRAIN_HYPERPARAMETERS,
                   model_hyper_parameters=DEFAULT_MODEL_HYPERPARAMETERS
                   ):
    # Start experiment
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

    # Transfer data to device
    datasets = move_data_to_device(datasets, device)

    # Create loggers
    train_logger = TrainLogMetrics(num_splits, ['supervised'])
    val_logger = TestLogMetrics(num_splits, ['accuracy', 'f1', 'f1_macro', 'f1_micro', 'auc_score', 'ap', 'pr_auc'])
    test_logger = TestLogMetrics(num_splits, ['accuracy', 'f1', 'f1_macro', 'f1_micro', 'auc_score', 'ap', 'pr_auc'])

    for run_id in range(num_splits):
        print(f'Split {run_id + 1}/{num_splits}')

        # Create the model
        model = create_model(model_hyper_parameters)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=train_hyperparameters["learning_rate"],
                                     weight_decay=train_hyperparameters["weight_decay"] if "weight_decay" in train_hyperparameters else None)
        BEST_VAL_METRIC = -np.inf
        best_model_path = interim_data_dir / f'model{run_id}.pth'

        early_stopping_cnt = 0
        num_epochs = train_hyperparameters['num_epochs']
        metric_to_optimize = train_hyperparameters['metric_to_optimize']
        for epoch in range(train_hyperparameters["num_epochs"]):
            if early_stopping_cnt > train_hyperparameters["early_stopping_limit"]:
                break
            model.train()
            optimizer.zero_grad()
            _, loss = model.loss()
            # Perform backpropagation
            loss.backward()
            optimizer.step()
            train_logger.train_update(run_id, 'supervised', loss.item())
            if epoch % train_hyperparameters["check_loss_freq"] == 0:
                # Validation step
                model.eval()
                with torch.no_grad():
                    val_metrics = eval_model(model,
                                             datasets[run_id]['train']['edges'],
                                             datasets[run_id]['val']['edges'],
                                             datasets[run_id]['train']['label'],
                                             datasets[run_id]['val']['label'])
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

        # Test performance
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        with torch.no_grad():
            test_metrics = eval_model(model,
                                      datasets[run_id]['train']['edges'],
                                      datasets[run_id]['test']['edges'],
                                      datasets[run_id]['train']['label'],
                                      datasets[run_id]['test']['label'])
            val_metrics = eval_model(model,
                                      datasets[run_id]['train']['edges'],
                                      datasets[run_id]['val']['edges'],
                                      datasets[run_id]['train']['label'],
                                      datasets[run_id]['val']['label'])
        for metric_name in test_metrics:
            test_logger.update(metric_name, run_id, test_metrics[metric_name])
            val_logger.update(metric_name, run_id, val_metrics[metric_name])

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

    # Simulation ended, report metrics on test set for the best model
    for metric_name in test_logger.test_metrics_dict:
        avg_val, std_val = test_logger.get_metric_stats(metric_name)
        print(f'Test {metric_name}: {avg_val}+-{std_val}')
        mlflow.log_metric(metric_name + '_avg', avg_val)
        mlflow.log_metric(metric_name + '_std', std_val)
        np.save(file=interim_data_dir / f'test_{metric_name}', arr=np.array(test_logger.test_metrics_dict[metric_name]))
        mlflow.log_artifact(interim_data_dir / f'test_{metric_name}.npy')
    # Simulation ended, report metrics on val set for the best model
    for metric_name in val_logger.test_metrics_dict:
        avg_val, std_val = val_logger.get_metric_stats(metric_name)
        print(f'Val {metric_name}: {avg_val}+-{std_val}')
        mlflow.log_metric('val_' + metric_name + '_avg', avg_val)
        mlflow.log_metric('val_' + metric_name + '_std', std_val)
        np.save(file=interim_data_dir / f'val_{metric_name}',
                arr=np.array(val_logger.test_metrics_dict[metric_name]))
        mlflow.log_artifact(interim_data_dir / f'val_{metric_name}.npy')
    return interim_data_dir


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
    device_id = '3'
    # General hyperparameters
    hyper_parameters = {'train_perc': train_perc, 'val_perc': val_perc, 'test_perc': test_perc,
                        'overwrite_data': overwrite_data}
    # optimization hyperparameters
    learning_rate = 0.001
    num_epochs = 1000
    check_loss_freq = 25
    early_stopping_limit = 10
    metric_to_optimize = 'auc_score'
    train_hyper_parameters = {'num_epochs': num_epochs,
                              'learning_rate': 1e-3,
                              'early_stopping_limit': early_stopping_limit,
                              'metric_to_optimize': metric_to_optimize,
                              'check_loss_freq': check_loss_freq,
                              }
    # model hyperparameters
    latent_dim = 64
    model_hyper_parameters = {'latent_dim': latent_dim}
    for seed_val in seed:
        mlflow.set_experiment(f'{metric_to_optimize}-{dataset_name}-MODEL-{seed_val}')
        for num_splits_val in num_splits:
            with mlflow.start_run():
                exp_dir = run_experiment(dataset_name=dataset_name,
                                         is_few_shot=is_few_shot,
                                         num_splits=num_splits_val,
                                         device_id=device_id,
                                         seed=seed_val,
                                         hyper_parameters=hyper_parameters,
                                         train_hyperparameters=train_hyper_parameters,
                                         model_hyper_parameters=model_hyper_parameters
                                         )
                try:
                    shutil.rmtree(exp_dir, ignore_errors=True)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))

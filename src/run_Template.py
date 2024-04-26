import os
import mlflow
import torch
import shutil
import numpy as np

from data_loader import create_data_loader
from data_processor import get_edge_index_with_sign
from model_eval import eval_model, TrainLogMetrics, TestLogMetrics
from my_utils import set_seed, setup_env, move_data_to_device
from models import MySDGNN
from plot_utils import plot_losses


# noinspection PyShadowingNames
def run_experiment(dataset_name='wiki',
                   overwrite_data=False,
                   learnable_features=False,
                   is_transductive=False,
                   random_masking=True,
                   unlabeled_perc=None,
                   val_perc=0.1,
                   test_perc=0.1,
                   mask_perc=0.05,
                   noise_perc=0.0,
                   seed=0,
                   num_splits=10,
                   device_id='',
                   learning_rate=0.001,
                   weight_decay=0.001,
                   num_epochs=1000,
                   check_loss_freq=25,
                   latent_dim=4,
                   num_layers=2,
                   metric_to_optimize='auc_score',
                   early_stopping_limit=10):
    # Start experiment
    # save parameters
    mlflow.log_param('dataset_name', dataset_name)
    mlflow.log_param('overwrite_data', overwrite_data)
    mlflow.log_param('learnable_features', learnable_features)
    mlflow.log_param('random_masking', random_masking)
    mlflow.log_param('val_perc', val_perc)
    mlflow.log_param('test_perc', test_perc)
    mlflow.log_param('noise_perc', noise_perc)
    mlflow.log_param('mask_perc', mask_perc)
    mlflow.log_param('unlabeled_perc', unlabeled_perc)
    mlflow.log_param('seed', seed)
    mlflow.log_param('num_splits', num_splits)
    mlflow.log_param('learning_rate', learning_rate)
    mlflow.log_param('weight_decay', weight_decay)
    mlflow.log_param('num_epochs', num_epochs)
    mlflow.log_param('latent_dim', latent_dim)
    mlflow.log_param('num_layers', num_layers)
    mlflow.log_param('metric_to_optimize', metric_to_optimize)
    mlflow.log_param('early_stopping_limit', early_stopping_limit)

    # set seed for reproducibility
    set_seed(seed)
    # set device
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    device, base_dir, interim_data_dir, data_dir = setup_env(device_id, dataset_name, seed, num_splits, val_perc,
                                                             test_perc, mask_perc, noise_perc,
                                                             filter_transitive=False,
                                                             is_transductive=is_transductive,
                                                             random_masking=random_masking)
    print(data_dir)
    # Create data loader for signed datasets
    signed_datasets = create_data_loader(dataset_name, base_dir, data_dir,
                                         hyper_params={'overwrite_data': overwrite_data,
                                                       'is_transductive': is_transductive,
                                                       'random_masking': random_masking,
                                                       'unlabeled_perc': unlabeled_perc,
                                                       'val_perc': val_perc,
                                                       'test_perc': test_perc,
                                                       'mask_perc': mask_perc,
                                                       'noise_perc': noise_perc,
                                                       'seed': seed,
                                                       'num_splits': num_splits,
                                                       'latent_dim': latent_dim,
                                                       'filter_transitive': False,
                                                       },
                                         device=device)

    # Transfer data to device
    signed_datasets = move_data_to_device(signed_datasets, device)

    # Create loggers
    train_logger = TrainLogMetrics(num_splits, ['supervised'])
    val_logger = TestLogMetrics(num_splits, ['accuracy', 'f1', 'f1_macro', 'f1_micro', 'auc_score', 'ap', 'pr_auc'])
    test_logger = TestLogMetrics(num_splits, ['accuracy', 'f1', 'f1_macro', 'f1_micro', 'auc_score', 'ap', 'pr_auc'])

    for run_id in range(num_splits):
        print(f'Split {run_id + 1}/{num_splits}')
        # Move validation and test labels to numpy for evaluation
        for split_type in ['val', 'test']:
            signed_datasets[run_id][split_type]['label'] = signed_datasets[run_id][split_type]['label'].detach().cpu().numpy()

        # reformat edge index to have for each edge an additional entry equal to the sign
        signed_datasets[run_id]['train']['label'] = signed_datasets[run_id]['train']['label'].float()
        edge_index_with_sign = get_edge_index_with_sign(signed_datasets[run_id]['train']['edges'],
                                                        signed_datasets[run_id]['train']['label']).long().to(device)
        # Create the model
        model = MySDGNN(node_num=signed_datasets[run_id]['features'].shape[0],
                        edge_index_s=edge_index_with_sign,
                        learnable_features=learnable_features,
                        in_dim=signed_datasets[run_id]['features'].shape[1],
                        out_dim=latent_dim,
                        layer_num=num_layers,
                        lamb_d=1.0,
                        lamb_t=0.0,
                        features=signed_datasets[run_id]['features'].to(device))
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
        BEST_VAL_METRIC = -np.inf
        best_model_path = interim_data_dir / f'model{run_id}.pth'

        early_stopping_cnt = 0
        for epoch in range(num_epochs):
            if early_stopping_cnt > early_stopping_limit:
                break
            model.train()
            optimizer.zero_grad()
            _, loss = model.loss()
            # Perform backpropagation
            loss.backward()
            optimizer.step()
            train_logger.train_update(run_id, 'supervised', loss.item())
            if epoch % check_loss_freq == 0:
                # Validation step
                model.eval()
                with torch.no_grad():
                    val_metrics = eval_model(model,
                                             signed_datasets[run_id]['train']['edges'],
                                             signed_datasets[run_id]['val']['edges'],
                                             signed_datasets[run_id]['train']['label'],
                                             signed_datasets[run_id]['val']['label'])
                    train_logger.val_update(run_id, val_metrics[metric_to_optimize])
                    if val_metrics[metric_to_optimize] > BEST_VAL_METRIC:
                        BEST_VAL_METRIC = val_metrics[metric_to_optimize]
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
                                      signed_datasets[run_id]['train']['edges'],
                                      signed_datasets[run_id]['test']['edges'],
                                      signed_datasets[run_id]['train']['label'],
                                      signed_datasets[run_id]['test']['label'])
            val_metrics = eval_model(model,
                                      signed_datasets[run_id]['train']['edges'],
                                      signed_datasets[run_id]['val']['edges'],
                                      signed_datasets[run_id]['train']['label'],
                                      signed_datasets[run_id]['val']['label'])
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
    dataset_name = 'slashdot'
    val_perc = 0.05
    test_perc = 0.05
    overwrite_data = False
    learnable_features = False
    is_transductive = False
    random_masking = True
    mask_perc = [0.75, ]
    noise_perc = [0.0, 0.1, 0.2]
    seed = [0, ]
    num_splits = [20, ]
    device_id = '3'
    # optimization hyperparameters
    learning_rate = 0.001
    weight_decay = 0.001
    num_epochs = 1000
    check_loss_freq = 25
    early_stopping_limit = 10
    # model hyperparameters
    latent_dim = 64
    num_layers = 2
    metric_to_optimize = 'accuracy'
    for seed_val in seed:
        mlflow.set_experiment(f'{metric_to_optimize}-{dataset_name}-SDGNN-{seed_val}')
        for noise_perc_val in noise_perc:
            for mask_perc_val in mask_perc:
                for num_splits_val in num_splits:
                    with mlflow.start_run():
                        exp_dir = run_experiment(dataset_name=dataset_name,
                                                 overwrite_data=overwrite_data,
                                                 learnable_features=learnable_features,
                                                 is_transductive=is_transductive,
                                                 random_masking=random_masking,
                                                 noise_perc=noise_perc_val,
                                                 val_perc=val_perc,
                                                 test_perc=test_perc,
                                                 mask_perc=mask_perc_val,
                                                 seed=seed_val,
                                                 num_splits=num_splits_val,
                                                 device_id=device_id,
                                                 learning_rate=learning_rate,
                                                 weight_decay=weight_decay,
                                                 num_epochs=num_epochs,
                                                 check_loss_freq=check_loss_freq,
                                                 latent_dim=latent_dim,
                                                 num_layers=num_layers,
                                                 metric_to_optimize=metric_to_optimize,
                                                 early_stopping_limit=early_stopping_limit
                                                 )
                        try:
                            shutil.rmtree(exp_dir, ignore_errors=True)
                        except OSError as e:
                            print("Error: %s - %s." % (e.filename, e.strerror))

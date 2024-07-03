import argparse
import os
import mlflow
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn

from my_utils import set_seed, setup_env, move_data_to_device, update_best_model_snapshot, save_metrics
from llm_utils import TweetDataset
from data_loader import create_data_loader
from model_eval import TrainLogMetrics, TestLogMetrics, eval_pred
from plot_utils import plot_losses

DEFAULT_HYPERPARAMETERS = {'train_perc': .6,
                           'val_perc': .2,
                           'test_perc': .2,
                           'num_splits': 5,
                           'aggr_type': 'mean'}
DEFAULT_TRAIN_HYPERPARAMETERS = {'input_embed': 'positional', 'epochs': 1000, 'learning_rate': 1e-3,
                                 'early_stopping_limit': 10, 'check_loss_freq': 5}
DEFAULT_MODEL_HYPERPARAMETERS = {}


def main(dataset_name, train_hyperparams, model_hyperparams, hyper_params, device_id):
    # Start experiment
    if model_hyperparams is None:
        model_hyperparams = DEFAULT_MODEL_HYPERPARAMETERS
    if train_hyperparams is None:
        train_hyperparams = DEFAULT_TRAIN_HYPERPARAMETERS
    if hyper_params is None:
        hyper_params = DEFAULT_HYPERPARAMETERS
    batch_size = train_hyperparams['batch_size']
    # set seed for reproducibility
    set_seed(hyper_params['seed'])
    # set device
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    device, base_dir, interim_data_dir, data_dir = setup_env(device_id, dataset_name, hyper_params)
    print(data_dir)
    # Create data loader for signed datasets
    datasets = create_data_loader(data_dir, hyper_params['tsim_th'])
    # Transfer data to device
    datasets = move_data_to_device(datasets, device)
    # Read tweets
    control_df = pd.read_csv(data_dir / 'CONTROL_mostPop_tweet_texts.csv', index_col=0)
    io_drivers_df = pd.read_csv(data_dir / 'IO_mostPop_tweet_texts.csv', index_col=0)
    merged_df = pd.concat([control_df, io_drivers_df])
    nodes_list = list(datasets['graph'].nodes())
    nodes_list_raw_fmt = list(map(lambda x: np.int64(datasets['noderemapping_rev'][x]), nodes_list))
    node_labels = datasets['labels']
    # Create loggers
    train_logger = TrainLogMetrics(hyper_params['num_splits'], ['supervised'])
    val_logger = TestLogMetrics(hyper_params['num_splits'],
                                ['accuracy', 'precision', 'f1_macro', 'f1_micro', 'roc_auc'])
    test_logger = TestLogMetrics(hyper_params['num_splits'],
                                 ['accuracy', 'precision', 'f1_macro', 'f1_micro', 'roc_auc'])
    # get training hyperparameters
    num_epochs = train_hyperparams['num_epochs']
    metric_to_optimize = train_hyperparams['metric_to_optimize']
    for run_id in tqdm(range(hyper_params['num_splits']), 'Splits training'):
        train_mask = datasets['splits'][run_id]['train']
        val_mask = datasets['splits'][run_id]['val']
        test_mask = datasets['splits'][run_id]['test']
        # Datasets
        train_dataset = TweetDataset(merged_df, nodes_list_raw_fmt, node_labels, train_mask, device)
        val_dataset = TweetDataset(merged_df, nodes_list_raw_fmt, node_labels, val_mask, device)
        excluded_users_df = datasets['excluded_users']
        excluded_users = excluded_users_df.userid.unique().tolist()
        test_users = nodes_list_raw_fmt + excluded_users
        test_mask_enhanced = np.concatenate([test_mask, np.full((len(excluded_users),), fill_value=True)])
        node_labels_enhanced = torch.cat([node_labels, torch.ones((len(excluded_users),)).to(device)])
        test_dataset = TweetDataset(merged_df, test_users, node_labels_enhanced, test_mask_enhanced, device)

        # Dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Define a simple classifier on top of SentenceTransformer embeddings
        class TweetClassifier(nn.Module):
            def __init__(self, input_dim):
                super(TweetClassifier, self).__init__()
                self.fc = nn.Linear(input_dim, 1)

            def forward(self, x):
                return torch.sigmoid(self.fc(x))

        BEST_VAL_METRIC = -np.inf
        best_model_path = interim_data_dir / f'model{run_id}.pth'
        # Create the model
        model = TweetClassifier(input_dim=768)  # Sentence-Transformer output dimension
        model.to(device)
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=train_hyperparams['learning_rate'])
        loss_fn = torch.nn.BCELoss()
        early_stopping_cnt = 0
        for epoch in range(num_epochs):
            if early_stopping_cnt > train_hyperparams["early_stopping_limit"]:
                break
            model.train()
            running_loss = 0.0
            for inputs, labels in train_dataloader:
                optimizer.zero_grad()
                outputs = model(inputs.float())
                loss = loss_fn(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_logger.train_update(run_id, 'supervised', running_loss / len(train_dataloader))
            if epoch % train_hyperparams["check_loss_freq"] == 0:
                # Validation step
                model.eval()
                with torch.no_grad():
                    val_pred, val_labels = [], []
                    for inputs, labels in val_dataloader:
                        outputs = model(inputs.float())
                        tmp_preds = outputs.squeeze().round().cpu().detach().numpy().flatten()
                        val_pred.extend(tmp_preds)
                        val_labels.extend(labels.cpu().detach().numpy())
                    val_metrics = eval_pred(np.array(val_labels), np.array(val_pred) > 0.5,
                                            np.full((len(val_pred),), fill_value=True), prob_pred=np.array(val_pred))
                    train_logger.val_update(run_id, val_metrics[train_hyperparams["metric_to_optimize"]])
                    if val_metrics[train_hyperparams["metric_to_optimize"]] > BEST_VAL_METRIC:
                        BEST_VAL_METRIC = val_metrics[train_hyperparams["metric_to_optimize"]]
                        torch.save(model.state_dict(), best_model_path)
                        early_stopping_cnt = 0
                    else:
                        early_stopping_cnt += 1
                    print(
                        f'Epoch {epoch}/{num_epochs} train_loss: {running_loss} -- val_{metric_to_optimize}: {val_metrics[metric_to_optimize]}')
            else:
                train_logger.val_update(run_id, 0.0)
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        with torch.no_grad():
            test_pred, test_labels = [], []
            for inputs, labels in test_dataloader:
                outputs = model(inputs.float())
                tmp_preds = outputs.squeeze().round().cpu().detach().numpy().flatten()
                test_pred.extend(tmp_preds)
                test_labels.extend(labels.cpu().detach().numpy())
            test_metrics = eval_pred(np.array(test_labels), np.array(test_pred) > 0.5,
                                     np.full((len(test_pred),), fill_value=True), prob_pred=np.array(test_pred))
        with torch.no_grad():
            val_pred, val_labels = [], []
            for inputs, labels in val_dataloader:
                outputs = model(inputs.float())
                tmp_preds = outputs.squeeze().round().cpu().detach().numpy().flatten()
                val_pred.extend(tmp_preds)
                val_labels.extend(labels.cpu().detach().numpy())
            val_metrics = eval_pred(np.array(val_labels), np.array(val_pred) > 0.5,
                                    np.full((len(val_pred),), fill_value=True), prob_pred=np.array(val_pred))
        # Evaluate perfomance on val set
        for metric_name in val_metrics:
            val_logger.update(metric_name, run_id, val_metrics[metric_name])
        # Evaluate perfomance on test set
        for metric_name in test_metrics:
            test_logger.update(metric_name, run_id, test_metrics[metric_name])

    for split_num in tqdm(range(hyper_params['num_splits']), 'Splits post-training'):
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
    update_best_model_snapshot(data_dir, metric_to_optimize, test_logger, hyper_params['num_splits'], interim_data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SBERT model")
    parser.add_argument('-dataset_name', '--dataset', type=str, help='Dataset', default='cuba')
    parser.add_argument('-seed', '--seed', type=int, help='Random seed', default=12121995)
    parser.add_argument('-train_perc', '--train', type=float, help='Training percentage', default=.6)
    parser.add_argument('-val_perc', '--val', type=float, help='Validation percentage', default=.2)
    parser.add_argument('-test_perc', '--test', type=float, help='Test percentage', default=.2)
    parser.add_argument('-num_splits', '--splits', type=int, help='Num of train-val-test splits', default=5)
    parser.add_argument('-tweet_sim_threshold', '--tsim_th', type=float, help='Threshold over which we retain an edge '
                                                                              'in tweet similarity network',
                        default=.99)
    # parser.add_argument('-heterogeneous', '--het', action='store_true', help="If True, return all the networks "
    #                                                                          "otherwise return the fused")
    parser.add_argument('-device_id', '--device', type=str, help='GPU ID#', default='1')
    parser.add_argument('-num_epochs', '--epochs', type=int, help='#Training Epochs', default=1000)
    parser.add_argument('-learning_rate', '--lr', type=float, help='Optimizer Learning Rate', default=1e-2)
    parser.add_argument('-batch_size', '--bs', type=int, help='Batch size', default=128)
    parser.add_argument('-early_stopping_limit', '--early', type=int, help='Num patience steps', default=20)
    parser.add_argument('-check_loss_freq', '--check', type=int, help='Frequency validation check', default=1)
    parser.add_argument('-metric_to_optimize', '--val_metric', type=str, help='Metric to optimize', default='roc_auc')
    args = parser.parse_args()
    # General hyperparameters
    hyper_parameters = {'train_perc': args.train, 'val_perc': args.val, 'test_perc': args.test,
                        'num_splits': args.splits, 'seed': args.seed,
                        'tsim_th': args.tsim_th}
    # optimization hyperparameters
    train_hyperparameters = {'num_epochs': args.epochs, 'learning_rate': args.lr,
                             'early_stopping_limit': args.early, 'check_loss_freq': args.check,
                             'metric_to_optimize': args.val_metric, 'batch_size': args.bs}
    # model hyperparameters
    model_hyperparameters = {}
    main(args.dataset, train_hyperparameters, model_hyperparameters, hyper_parameters, args.device)

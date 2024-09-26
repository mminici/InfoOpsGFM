import argparse
import os
import mlflow
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from models import GNN
from my_utils import set_seed, setup_env, move_data_to_device, update_best_model_snapshot, set_maximum_edge_weights \
    , save_metrics, get_edge_index, tensors_from_ids, handle_isolated_nodes, get_gnn_embeddings
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
DEFAULT_MODEL_HYPERPARAMETERS = {'gnn_type': 'gcn', 'latent_dim': 32, 'dropout': 0.2}


def create_model(model_hyperparams):
    class GNN_CrossAttention(torch.nn.Module):
        def __init__(self, num_node_features, hidden_dim, num_classes, num_textual_features, num_structural_features,
                     activation_fn=torch.nn.ReLU(), dropout_p=0.2, gnn_type='gcn'):
            super().__init__()
            self.gnn = GNN(num_node_features=num_node_features * 2,
                           hidden_dim=hidden_dim * 2, num_classes=num_classes,
                           dropout_p=dropout_p, gnn_type=gnn_type)
            self.cross_attention_to_text = torch.nn.Linear(num_structural_features, hidden_dim)
            self.cross_attention_to_struct = torch.nn.Linear(num_textual_features, hidden_dim)
            self.struct_projector = torch.nn.Sequential(torch.nn.Linear(num_structural_features, hidden_dim),
                                                        torch.nn.ReLU())
            self.text_projector = torch.nn.Sequential(torch.nn.Linear(num_textual_features, hidden_dim),
                                                      torch.nn.ReLU())
            self.joint_projector = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * 2, hidden_dim * 2),
                torch.nn.ReLU(),
                # nn.Linear(hidden_dim * 2, hidden_dim * 2),
                # nn.ReLU()
            )

        def forward(self, text_node_features, struct_node_features, edge_index):
            struct_projection = self.struct_projector(struct_node_features) * self.cross_attention_to_struct(text_node_features)
            text_projection = self.text_projector(text_node_features) * self.cross_attention_to_text(struct_node_features)
            multimodal_node_features = self.joint_projector(torch.concat([struct_projection, text_projection], dim=-1))
            return self.gnn(multimodal_node_features, edge_index)

    return GNN_CrossAttention(num_node_features=model_hyperparams['latent_dim'],
                              hidden_dim=model_hyperparams['latent_dim'], num_classes=2,
                              dropout_p=model_hyperparams['dropout'], gnn_type=model_hyperparams['gnn_type'],
                              num_textual_features=model_hyperparams['num_textual_features'],
                              num_structural_features=model_hyperparams['num_structural_features'])


def main(dataset_name, train_hyperparams, model_hyperparams, hyper_params, device_id):
    # Start experiment
    if model_hyperparams is None:
        model_hyperparams = DEFAULT_MODEL_HYPERPARAMETERS
    if train_hyperparams is None:
        train_hyperparams = DEFAULT_TRAIN_HYPERPARAMETERS
    if hyper_params is None:
        hyper_params = DEFAULT_HYPERPARAMETERS
    # set seed for reproducibility
    set_seed(hyper_params['seed'])
    # set device
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    device, base_dir, interim_data_dir, data_dir = setup_env(device_id, dataset_name, hyper_params)
    print('binary_IOHunter')
    print(data_dir)
    # Create data loader for signed datasets
    datasets = create_data_loader(data_dir, hyper_params['tsim_th'],
                                  hyper_params['train_perc'], hyper_params['undersampling_perc'],
                                  hyper_params['undersampling'])
    # Transfer data to device
    datasets = move_data_to_device(datasets, device)
    datasets['graph'] = set_maximum_edge_weights(datasets['graph'], [datasets['coRT'], datasets['coURL'],
                                                                     datasets['hashSeq'], datasets['tweetSim']])
    _, network = handle_isolated_nodes(datasets['graph'])
    # Get edge index representation
    print('Get edge index from graph ({}N {}E)'.format(network.number_of_nodes(),
                                                       network.number_of_edges()))
    edge_index = get_edge_index(network, data_dir)
    edge_index = edge_index.to(device)
    # Get node features
    print('Computing LLM-based features...')
    # Read tweets
    num_mostPop = hyper_params['most_pop']
    control_df = pd.read_csv(data_dir / f'CONTROL_mostPop{num_mostPop}_tweet_texts.csv', index_col=0)
    iodrivers_df = pd.read_csv(data_dir / f'IO_mostPop{num_mostPop}_tweet_texts.csv', index_col=0)
    merged_df = pd.concat([control_df, iodrivers_df])
    if (data_dir / f'sbert_nodeattributes_mostPop{num_mostPop}.pt').exists():
        node_features = torch.load(data_dir / f'sbert_nodeattributes_mostPop{num_mostPop}.pt')
    else:
        nodes_list = list(datasets['graph'].nodes())
        nodes_list_raw_fmt = list(map(lambda x: np.int64(datasets['noderemapping_rev'][x]), nodes_list))
        node_labels = datasets['labels']
        tweet_dataset = TweetDataset(merged_df, nodes_list_raw_fmt, node_labels,
                                     np.array([True] * len(nodes_list_raw_fmt)),
                                     device)
        node_features = tensors_from_ids(tweet_dataset.user_embeddings, nodes_list_raw_fmt)
        torch.save(node_features, data_dir / f'sbert_nodeattributes_mostPop{num_mostPop}.pt')
    node_features = node_features.to(device)
    print('Computing GNN features ({})...'.format(train_hyperparams['input_embed']))
    struct_node_features = get_gnn_embeddings(data_dir, {'type': train_hyperparams['input_embed'],
                                                         'trace_type': hyper_params['trace_type'],
                                                         'latent_dim': model_hyperparams['latent_dim'],
                                                         'seed': hyper_params['seed'],
                                                         'num_nodes': network.number_of_nodes(),
                                                         'graph': network, 'device': device,
                                                         'dataset_name': dataset_name, 'base_dir': base_dir,
                                                         'num_cores': 8,
                                                         'aggr_type': hyper_params['aggr_type']})
    struct_node_features = struct_node_features.to(device)
    model_hyperparams['num_textual_features'] = node_features.shape[1]
    model_hyperparams['num_structural_features'] = struct_node_features.shape[1]
    # Create loggers
    train_logger = TrainLogMetrics(hyper_params['num_splits'], ['supervised'])
    val_logger = TestLogMetrics(hyper_params['num_splits'], ['accuracy', 'precision', 'f1_macro', 'f1_micro'])
    test_logger = TestLogMetrics(hyper_params['num_splits'], ['accuracy', 'precision', 'f1_macro', 'f1_micro'])
    test_logger_coRT = TestLogMetrics(hyper_params['num_splits'],
                                      ['accuracy', 'precision', 'f1_macro', 'f1_micro', 'roc_auc'])
    test_logger_coURL = TestLogMetrics(hyper_params['num_splits'],
                                       ['accuracy', 'precision', 'f1_macro', 'f1_micro', 'roc_auc'])
    test_logger_hashSeq = TestLogMetrics(hyper_params['num_splits'],
                                         ['accuracy', 'precision', 'f1_macro', 'f1_micro', 'roc_auc'])
    # test_logger_fastRT = TestLogMetrics(hyper_params['num_splits'],
    #                                     ['accuracy', 'precision', 'f1_macro', 'f1_micro', 'roc_auc'])
    test_logger_tweetSim = TestLogMetrics(hyper_params['num_splits'],
                                          ['accuracy', 'precision', 'f1_macro', 'f1_micro', 'roc_auc'])
    coRT_mask = np.full(shape=(datasets['graph'].number_of_nodes(),), fill_value=False)
    coRT_mask[list(datasets['coRT'].nodes())] = True
    coURL_mask = np.full(shape=(datasets['graph'].number_of_nodes(),), fill_value=False)
    coURL_mask[list(datasets['coURL'].nodes())] = True
    hashSeq_mask = np.full(shape=(datasets['graph'].number_of_nodes(),), fill_value=False)
    hashSeq_mask[list(datasets['hashSeq'].nodes())] = True
    # fastRT_mask = np.full(shape=(datasets['graph'].number_of_nodes(),), fill_value=False)
    # fastRT_mask[list(datasets['fastRT'].nodes())] = True
    tweetSim_mask = np.full(shape=(datasets['graph'].number_of_nodes(),), fill_value=False)
    tweetSim_mask[list(datasets['tweetSim'].nodes())] = True
    # Create numpy version of labels for the validation phase
    numpy_labels = datasets['labels'].long().detach().cpu().numpy()
    # Get training hyperparameters
    num_epochs = train_hyperparams['num_epochs']
    metric_to_optimize = train_hyperparams['metric_to_optimize']
    for run_id in tqdm(range(hyper_params['num_splits']), 'Splits training'):
        BEST_VAL_METRIC = -np.inf
        best_model_path = interim_data_dir / f'model{run_id}.pth'
        # Create the model
        model = create_model(model_hyperparams)
        model.to(device)
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=train_hyperparams['learning_rate'])
        loss_fn = torch.nn.BCELoss()
        early_stopping_cnt = 0
        for epoch in range(num_epochs):
            if early_stopping_cnt > train_hyperparams["early_stopping_limit"]:
                break
            model.train()
            optimizer.zero_grad()
            pred = model(node_features, struct_node_features, edge_index).flatten()
            loss = loss_fn(pred[datasets['splits'][run_id]['train']],
                           datasets['labels'][datasets['splits'][run_id]['train']])
            loss.backward()
            optimizer.step()
            train_logger.train_update(run_id, 'supervised', loss.item())
            if epoch % train_hyperparams["check_loss_freq"] == 0:
                # Validation step
                model.eval()
                with torch.no_grad():
                    pred = model(node_features, struct_node_features, edge_index).detach().cpu().numpy().flatten()
                    val_metrics = eval_pred(numpy_labels, pred > 0.5, datasets['splits'][run_id]['val'])
                    train_logger.val_update(run_id, val_metrics[train_hyperparams["metric_to_optimize"]])
                    if val_metrics[train_hyperparams["metric_to_optimize"]] > BEST_VAL_METRIC:
                        BEST_VAL_METRIC = val_metrics[train_hyperparams["metric_to_optimize"]]
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
            pred = model(node_features, struct_node_features, edge_index).detach().cpu().numpy().flatten()
        # Evaluate perfomance on val set
        val_metrics = eval_pred(numpy_labels, pred > 0.5, datasets['splits'][run_id]['val'])
        for metric_name in val_metrics:
            val_logger.update(metric_name, run_id, val_metrics[metric_name])
        # Evaluate perfomance on test set
        test_metrics = eval_pred(numpy_labels, pred > 0.5, datasets['splits'][run_id]['test'])
        for metric_name in test_metrics:
            test_logger.update(metric_name, run_id, test_metrics[metric_name])
        # Evaluate perfomance on test set (only coRT nodes)
        test_metrics_coRT = eval_pred(numpy_labels, pred > 0.5,
                                      np.logical_and(datasets['splits'][run_id]['test'], coRT_mask),
                                      prob_pred=pred)
        for metric_name in test_metrics_coRT:
            test_logger_coRT.update(metric_name, run_id, test_metrics_coRT[metric_name])
        # Evaluate perfomance on test set (only coURL nodes)
        test_metrics_coURL = eval_pred(numpy_labels, pred > 0.5,
                                       np.logical_and(datasets['splits'][run_id]['test'], coURL_mask),
                                       prob_pred=pred)
        for metric_name in test_metrics_coURL:
            test_logger_coURL.update(metric_name, run_id, test_metrics_coURL[metric_name])
        # Evaluate perfomance on test set (only hashSeq nodes)
        test_metrics_hashSeq = eval_pred(numpy_labels, pred > 0.5,
                                         np.logical_and(datasets['splits'][run_id]['test'], hashSeq_mask),
                                         prob_pred=pred)
        for metric_name in test_metrics_hashSeq:
            test_logger_hashSeq.update(metric_name, run_id, test_metrics_hashSeq[metric_name])
        # Evaluate perfomance on test set (only fastRT nodes)
        # test_metrics_fastRT = eval_pred(numpy_labels, pred > 0.5,
        #                                 np.logical_and(datasets['splits'][run_id]['test'], fastRT_mask),
        #                                 prob_pred=pred)
        # for metric_name in test_metrics_fastRT:
        #     test_logger_fastRT.update(metric_name, run_id, test_metrics_fastRT[metric_name])
        # Evaluate perfomance on test set (only tweetSim nodes)
        test_metrics_tweetSim = eval_pred(numpy_labels, pred > 0.5,
                                          np.logical_and(datasets['splits'][run_id]['test'], tweetSim_mask),
                                          prob_pred=pred)
        for metric_name in test_metrics_tweetSim:
            test_logger_tweetSim.update(metric_name, run_id, test_metrics_tweetSim[metric_name])

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
    save_metrics(test_logger_coRT, interim_data_dir, 'TEST_coRT')
    save_metrics(test_logger_coURL, interim_data_dir, 'TEST_coURL')
    save_metrics(test_logger_hashSeq, interim_data_dir, 'TEST_hashSeq')
    # save_metrics(test_logger_fastRT, interim_data_dir, 'TEST_fastRT')
    save_metrics(test_logger_tweetSim, interim_data_dir, 'TEST_tweetSim')

    # Save best models
    update_best_model_snapshot(data_dir, metric_to_optimize, test_logger, hyper_params['num_splits'], interim_data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run GNN model")
    parser.add_argument('-dataset_name', '--dataset', type=str, help='Dataset', default='russia')
    parser.add_argument('-seed', '--seed', type=int, help='Random seed', default=12121995)
    parser.add_argument('-train_perc', '--train', type=float, help='Training percentage', default=.6)
    parser.add_argument('-val_perc', '--val', type=float, help='Validation percentage', default=.2)
    parser.add_argument('-test_perc', '--test', type=float, help='Test percentage', default=.2)
    parser.add_argument('-num_splits', '--splits', type=int, help='Num of train-val-test splits', default=5)
    parser.add_argument('-tweet_sim_threshold', '--tsim_th', type=float, help='Threshold over which we retain an edge '
                                                                              'in tweet similarity network',
                        default=.7)
    # parser.add_argument('-heterogeneous', '--het', action='store_true', help="If True, return all the networks "
    #                                                                          "otherwise return the fused")
    parser.add_argument('-device_id', '--device', type=str, help='GPU ID#', default='0')
    parser.add_argument('-gnn_aggr_fn', '--aggr_fn', type=str, help='GNN aggregation function', default='mean')
    parser.add_argument('-num_epochs', '--epochs', type=int, help='#Training Epochs', default=1000)
    parser.add_argument('-learning_rate', '--lr', type=float, help='Optimizer Learning Rate', default=1e-2)
    parser.add_argument('-early_stopping_limit', '--early', type=int, help='Num patience steps', default=20)
    parser.add_argument('-check_loss_freq', '--check', type=int, help='Frequency validation check', default=1)
    parser.add_argument('-metric_to_optimize', '--val_metric', type=str, help='Metric to optimize', default='f1_macro')
    parser.add_argument('-gnn_type', '--gnn', type=str, help='GNN Model type', default='gcn')
    parser.add_argument('-gnn_embed_type', '--embed_type', type=str, help='GNN Embedding Type', default='positional_degree')
    parser.add_argument('-latent_dim', '--latent', type=int, help='Latent dimension', default=128)
    parser.add_argument('-dropout', '--dropout', type=float, help='Dropout frequency', default=.2)
    parser.add_argument('-min_tweets', '--min_tweets', type=int,
                        help='Minimum number of tweets a user needs to have to be included in the dataset',
                        default=10)
    parser.add_argument('-most_popular', '--most_pop', type=int,
                        help='Number of most popular tweets to use to represent a user',
                        default=5)
    parser.add_argument('-under_sampling_perc', '--under_perc', help='undersampling percentage', default=None)
    parser.add_argument('-under_sampling', '--under', help='undersampling num labels', default=None)
    args = parser.parse_args()
    # General hyperparameters
    hyper_parameters = {'train_perc': args.train, 'val_perc': args.val, 'test_perc': args.test,
                        'aggr_type': args.aggr_fn, 'num_splits': args.splits, 'seed': args.seed,
                        'tsim_th': args.tsim_th,
                        'min_tweets': args.min_tweets, 'most_pop': args.most_pop,
                        'input_embed': args.embed_type, 'trace_type': 'all',
                        'undersampling_perc': float(args.under_perc) if args.under_perc is not None else None,
                        'undersampling': int(args.under) if args.under is not None else None
                        }
    # optimization hyperparameters
    train_hyperparameters = {'num_epochs': args.epochs, 'learning_rate': args.lr,
                             'early_stopping_limit': args.early, 'check_loss_freq': args.check,
                             'metric_to_optimize': args.val_metric,
                             'input_embed': args.embed_type, 'trace_type': 'all'
                             }
    # model hyperparameters
    model_hyperparameters = {'gnn_type': args.gnn, 'latent_dim': args.latent, 'dropout': args.dropout}
    main(args.dataset, train_hyperparameters, model_hyperparameters, hyper_parameters, args.device)

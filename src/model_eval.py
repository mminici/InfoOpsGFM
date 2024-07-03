import numpy as np

from sklearn import metrics
from tqdm import tqdm


def eval_pred(ground_truth, pred, mask, prob_pred=None):
    if mask.sum() > 0:
        metrics_dict = {'f1_macro': metrics.f1_score(ground_truth[mask], pred[mask], average='macro'),
                        'f1_micro': metrics.f1_score(ground_truth[mask], pred[mask], average='micro'),
                        'accuracy': metrics.accuracy_score(ground_truth[mask], pred[mask]),
                        'precision': metrics.precision_score(ground_truth[mask], pred[mask])}
        if prob_pred is not None:
            metrics_dict['roc_auc'] = metrics.roc_auc_score(ground_truth[mask], prob_pred[mask])
    else:
        metrics_dict = {'f1_macro': None,
                        'f1_micro': None,
                        'accuracy': None,
                        'precision': None,
                        'roc_auc': None}
    return metrics_dict


def get_best_threshold(ground_truth, pred_by_threshold, mask, metric_to_optimize):
    assert metric_to_optimize in ['f1_macro', 'f1_micro', 'accuracy', 'precision']
    metrics_dict = {'f1_macro': [],
                    'f1_micro': [],
                    'accuracy': [],
                    'precision': []}
    for i in tqdm(range(len(pred_by_threshold))):
        metrics_dict['f1_macro'].append(metrics.f1_score(ground_truth[mask],
                                                         pred_by_threshold[i][mask], average='macro'))
        metrics_dict['f1_micro'].append(metrics.f1_score(ground_truth[mask],
                                                         pred_by_threshold[i][mask], average='micro'))
        metrics_dict['accuracy'].append(metrics.accuracy_score(ground_truth[mask],
                                                               pred_by_threshold[i][mask]))
        metrics_dict['precision'].append(metrics.precision_score(ground_truth[mask],
                                                                 pred_by_threshold[i][mask]))
    return np.argmax(metrics_dict[metric_to_optimize])


class TrainLogMetrics:
    def __init__(self, num_splits, loss_types_list):
        self.train_loss_dict = {}
        self.val_metrics_dict = {}
        for run_id in range(num_splits):
            self.train_loss_dict[run_id] = {loss_type: [] for loss_type in loss_types_list}
            self.val_metrics_dict[run_id] = []

    def train_update(self, run_id, loss_type, value):
        self.train_loss_dict[run_id][loss_type].append(value)

    def val_update(self, run_id, value):
        self.val_metrics_dict[run_id].append(value)


class TestLogMetrics:
    def __init__(self, num_splits, metric_names_list):
        self.test_metrics_dict = {}
        for metric in metric_names_list:
            self.test_metrics_dict[metric] = [None] * num_splits

    def update(self, metric_name, run_id, value):
        self.test_metrics_dict[metric_name][run_id] = value

    def get_metric_stats(self, metric_name, float_precision=4):
        avg_val, std_val = np.mean(self.test_metrics_dict[metric_name]), np.std(self.test_metrics_dict[metric_name])
        return round(avg_val, float_precision), round(std_val, float_precision)

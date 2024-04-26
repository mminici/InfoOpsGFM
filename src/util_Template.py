import pathlib
import uuid

import torch


def setup_env(device_id, dataset_name, seed, num_splits, is_few_shot, hyper_parameters):
    device = torch.device("cuda" if torch.cuda.is_available() and device_id != "-1" else "cpu")
    # Creating folder to host run-specific files
    base_dir = pathlib.Path.cwd().parent
    my_run_id = uuid.uuid4()
    interim_data_dir = base_dir / 'data' / 'interim' / f"{my_run_id}"
    interim_data_dir.mkdir(exist_ok=True, parents=True)
    # Import dataset
    processed_data_dir = base_dir / 'data' / 'processed'
    data_dir = processed_data_dir / dataset_name
    data_dir = data_dir / f'seed_{seed}_num_splits_{num_splits}'
    if is_few_shot:
        num_few_shot_train = hyper_parameters['num_few_shot_train']
        num_few_shot_val = hyper_parameters['num_few_shot_val']
        test_perc = hyper_parameters['test_perc']
        data_dir = data_dir / 'fsl' / f'train_{num_few_shot_train}_val_{num_few_shot_val}_test_{round(test_perc, 2)}'
        data_dir.mkdir(exist_ok=True, parents=True)
        return device, base_dir, interim_data_dir, data_dir
    train_perc = hyper_parameters['train_perc']
    val_perc = hyper_parameters['val_perc']
    test_perc = hyper_parameters['test_perc']
    data_dir = data_dir / f'train_{round(train_perc, 2)}_val_{round(val_perc, 2)}_test_{round(test_perc, 2)}'
    data_dir.mkdir(exist_ok=True, parents=True)
    return device, base_dir, interim_data_dir, data_dir

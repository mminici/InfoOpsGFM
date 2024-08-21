import pickle

DATASET_FILENAME = 'datasets.pkl'


def load_dataset(data_dir, filter_th, tr_perc, undersampling):
    fname = f'{filter_th}_{DATASET_FILENAME}'
    if tr_perc != 0.6:
        fname = f'{filter_th}_{DATASET_FILENAME}_{tr_perc}'
    if undersampling is not None:
        fname += f'_{undersampling}U'
    with open(data_dir / fname, 'rb') as file:
        return pickle.load(file)


def create_data_loader(data_dir, filter_th, tr_perc, undersampling):
    datasets = load_dataset(data_dir, filter_th, tr_perc, undersampling)
    return datasets

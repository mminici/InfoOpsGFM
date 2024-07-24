import pickle

DATASET_FILENAME = 'datasets.pkl'
# Script constants
CONTROL_FILE_IDX, IO_FILE_IDX = 0, 1
filename_dict = {'UAE_sample': ['control_driver_tweets_uae_082019.jsonl', 'uae_082019_tweets_csv_unhashed.csv'],
                 'cuba': ['control_driver_tweets_cuba_082020.jsonl', 'cuba_082020_tweets_csv_unhashed.csv']}


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

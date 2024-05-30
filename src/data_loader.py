import pickle

DATASET_FILENAME = 'datasets.pkl'
# Script constants
CONTROL_FILE_IDX, IO_FILE_IDX = 0, 1
filename_dict = {'UAE_sample': ['control_driver_tweets_uae_082019.jsonl', 'uae_082019_tweets_csv_unhashed.csv'],
                 'cuba': ['control_driver_tweets_cuba_082020.jsonl', 'cuba_082020_tweets_csv_unhashed.csv']}


def load_dataset(data_dir, filter_th):
    with open(data_dir / f'{filter_th}_{DATASET_FILENAME}', 'rb') as file:
        return pickle.load(file)


def create_data_loader(data_dir, filter_th):
    datasets = load_dataset(data_dir, filter_th)
    return datasets

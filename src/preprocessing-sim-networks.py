import argparse
import pathlib
import pickle
import pandas as pd
import networkx as nx

import preprocessing_util
import tweetSimUtil


def save_network(network, path):
    with open(path, 'wb') as f:
        pickle.dump(network, f)


CONTROL_USERS_FILENAME = {'UAE_sample': 'control_driver_tweets_uae_082019.jsonl',
                          'cuba': 'control_driver_tweets_cuba_082020.jsonl'}
IO_USERS_FILENAME = {'UAE_sample': 'uae_082019_tweets_csv_unhashed.csv',
                     'cuba': 'cuba_082020_tweets_csv_unhashed.csv'}


def main(dataset_name):
    # Basic definitions
    base_dir = pathlib.Path.cwd().parent
    processed_datadir = base_dir / 'data' / 'processed'
    data_dir = processed_datadir / dataset_name
    data_dir.mkdir(parents=True, exist_ok=True)

    print('Importing control file...')
    print(base_dir / 'data' / 'raw' / dataset_name / CONTROL_USERS_FILENAME[dataset_name])
    control_df = pd.read_json(base_dir / 'data' / 'raw' / dataset_name / CONTROL_USERS_FILENAME[dataset_name],
                              lines=True)
    print('Importing IO drivers file...')
    print(base_dir / 'data' / 'raw' / dataset_name / IO_USERS_FILENAME[dataset_name])
    iodrivers_df = pd.read_csv(base_dir / 'data' / 'raw' / dataset_name / IO_USERS_FILENAME[dataset_name], sep=",")
    print('Get CoRetweet network...')
    coRT = preprocessing_util.coRetweet(control_df, iodrivers_df)
    save_network(coRT, data_dir / 'coRT.pkl')
    print('Get CoURL network...')
    coURL = preprocessing_util.coURL(control_df, iodrivers_df)
    save_network(coURL, data_dir / 'coURL.pkl')
    print('Get HashtagSeq network...')
    hashSeq = preprocessing_util.hashSeq(control_df, iodrivers_df, minHashtags=5)
    save_network(hashSeq, data_dir / 'hashSeq.pkl')
    print('Get fastRetweet network...')
    fastRT = preprocessing_util.fastRetweet(control_df, iodrivers_df, timeInterval=10)
    save_network(fastRT, data_dir / 'fastRT.pkl')
    print('Get tweetSimilarity network...')
    tweetSimPath = data_dir / 'tweetSim'
    tweetSimPath.mkdir(parents=True, exist_ok=True)
    tweetSim = tweetSimUtil.getTweetSimNetwork(control_df, iodrivers_df, outputDir=tweetSimPath)
    save_network(tweetSim, data_dir / 'tweetSim.pkl')
    print('Deriving fused network...')
    fusedNet = coRT.copy()
    fusedNet = nx.compose(fusedNet, coURL)
    fusedNet = nx.compose(fusedNet, hashSeq)
    fusedNet = nx.compose(fusedNet, fastRT)
    fusedNet = nx.compose(fusedNet, tweetSim)
    save_network(fusedNet, data_dir / 'fusedNet.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess dataset to get all similarity networks")
    parser.add_argument('-dataset_name', '--dataset', type=str, help='Dataset')
    # parser.add_argument('-f', '--flag', action='store_true', help='A boolean flag')
    args = parser.parse_args()
    main(args.dataset_name)

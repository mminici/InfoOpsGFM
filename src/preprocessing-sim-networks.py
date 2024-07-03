import argparse
import pathlib
import pickle
import pandas as pd
import networkx as nx

import preprocessing_util
import tweetSimUtil


# Function to check if a node ID is a valid numeric string
def is_valid_node_id(node_id):
    try:
        # Check if the node ID is a string representation of a number
        return isinstance(node_id, str) and node_id.isdigit()
    except:
        return False


def correct_nodeIDs(G):
    # Initial count of valid node IDs
    valid_node_ids = [node for node in G.nodes if is_valid_node_id(node)]
    valid_node_count = len(valid_node_ids)

    # Remove invalid node IDs
    invalid_node_ids = [node for node in G.nodes if not is_valid_node_id(node)]
    # G.remove_nodes_from(invalid_node_ids)
    print('Valid', valid_node_count)
    print('Invalid', len(invalid_node_ids))
    corrected_G = G.copy()
    corrected_G.remove_nodes_from(invalid_node_ids)
    print('Valid+Invalid #edges', G.number_of_edges())
    print('Valid #edges', corrected_G.number_of_edges())
    return corrected_G


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
    coRT = correct_nodeIDs(coRT)
    save_network(coRT, data_dir / 'coRT.pkl')
    print('Get CoURL network...')
    coURL = preprocessing_util.coURL(control_df, iodrivers_df)
    coURL = correct_nodeIDs(coURL)
    save_network(coURL, data_dir / 'coURL.pkl')
    print('Get HashtagSeq network...')
    hashSeq = preprocessing_util.hashSeq(control_df, iodrivers_df, minHashtags=5)
    hashSeq = correct_nodeIDs(hashSeq)
    save_network(hashSeq, data_dir / 'hashSeq.pkl')
    print('Get fastRetweet network...')
    fastRT = preprocessing_util.fastRetweet(control_df, iodrivers_df, timeInterval=10)
    fastRT = correct_nodeIDs(fastRT)
    save_network(fastRT, data_dir / 'fastRT.pkl')
    print('Get tweetSimilarity network...')
    tweetSimPath = data_dir / 'tweetSim'
    tweetSimPath.mkdir(parents=True, exist_ok=True)
    tweetSim = tweetSimUtil.getTweetSimNetwork(control_df, iodrivers_df, outputDir=tweetSimPath)
    tweetSim = correct_nodeIDs(tweetSim)
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

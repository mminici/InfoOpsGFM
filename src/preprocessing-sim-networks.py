import argparse
import pathlib
import pickle
import numpy as np
import networkx as nx

import preprocessing_util
import tweetSimUtil
import my_utils


# Function to check if a node ID is a valid numeric string
def is_valid_node_id(node_id):
    try:
        # Check if the node ID is a string representation of a number
        return isinstance(node_id, str) and node_id.isdigit()
    except:
        return False


def is_numeric(value):
    """
    Check if the given value is numeric (either integer or float).

    Parameters:
    value (str): The input string to check.

    Returns:
    bool: True if the value is numeric, False otherwise.
    """
    try:
        float(value)  # Try to convert the value to a float
        return True
    except ValueError:
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


def main(dataset_name, device_id):
    # Basic definitions
    base_dir = pathlib.Path.cwd().parent
    processed_datadir = base_dir / 'data' / 'processed'
    data_dir = processed_datadir / dataset_name
    data_dir.mkdir(parents=True, exist_ok=True)

    print('Importing control file...')
    print(base_dir / 'data' / 'raw' / dataset_name / f'{dataset_name}_tweets_control.pkl.gz')
    control_df = my_utils.read_compressed_pickle(
        base_dir / 'data' / 'raw' / dataset_name / f'{dataset_name}_tweets_control.pkl.gz')
    print('Importing IO drivers file...')
    print(base_dir / 'data' / 'raw' / dataset_name / f'{dataset_name}_tweets_io.pkl.gz')
    iodrivers_df = my_utils.read_compressed_pickle(
        base_dir / 'data' / 'raw' / dataset_name / f'{dataset_name}_tweets_io.pkl.gz')
    # Filter out any node ID that is not numeric/valid
    control_df = control_df[control_df['userid'].apply(lambda x: is_numeric(x))]
    iodrivers_df = iodrivers_df[iodrivers_df['userid'].apply(lambda x: is_numeric(x))]
    preprocessing_module = preprocessing_util
    tweetSim_module = tweetSimUtil
    print('Get CoRetweet network...')
    coRT = preprocessing_module.coRetweet(control_df, iodrivers_df)
    # coRT = correct_nodeIDs(coRT)
    save_network(coRT, data_dir / 'coRT.pkl')
    print('Get CoURL network...')
    coURL = preprocessing_module.coURL(control_df, iodrivers_df)
    # coURL = correct_nodeIDs(coURL)
    save_network(coURL, data_dir / 'coURL.pkl')
    print('Get HashtagSeq network...')
    hashSeq = preprocessing_module.hashSeq(control_df, iodrivers_df, minHashtags=5)
    # hashSeq = correct_nodeIDs(hashSeq)
    save_network(hashSeq, data_dir / 'hashSeq.pkl')
    # print('Get fastRetweet network...')
    # fastRT = preprocessing_module.fastRetweet(control_df, iodrivers_df, timeInterval=10)
    # fastRT = correct_nodeIDs(fastRT)
    # save_network(fastRT, data_dir / 'fastRT.pkl')
    print('Get tweetSimilarity network...')
    tweetSimPath = data_dir / 'tweetSim'
    tweetSimPath.mkdir(parents=True, exist_ok=True)
    # Applying node filtering
    # At the moment, we exclude all users having less than 10 tweets
    control_df['userid'] = control_df['userid'].apply(lambda x: np.int64(x))
    iodrivers_df['userid'] = iodrivers_df['userid'].apply(lambda x: np.int64(x))
    # Grouping by 'userid' and filtering those with at least 5 rows
    io_drivers_userids = iodrivers_df.groupby('userid').filter(lambda x: len(x) >= 10)[
        'userid'].unique()
    control_userids = control_df.groupby('userid').filter(lambda x: len(x) >= 10)['userid'].unique()
    control_df = control_df[control_df.userid.isin(control_userids)]
    iodrivers_df = iodrivers_df[iodrivers_df.userid.isin(io_drivers_userids)]
    tweetSim = tweetSim_module.getTweetSimNetwork(control_df, iodrivers_df, timeWindow=1,
                                                  outputDir=tweetSimPath, cudaId=device_id)
    # tweetSim = correct_nodeIDs(tweetSim)
    save_network(tweetSim, data_dir / 'tweetSim.pkl')
    print('Deriving fused network...')
    fusedNet = coRT.copy()
    fusedNet = nx.compose(fusedNet, coURL)
    fusedNet = nx.compose(fusedNet, hashSeq)
    # fusedNet = nx.compose(fusedNet, fastRT)
    fusedNet = nx.compose(fusedNet, tweetSim)
    save_network(fusedNet, data_dir / 'fusedNet.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess dataset to get all similarity networks")
    parser.add_argument('-dataset_name', '--dataset', type=str, help='Dataset')
    parser.add_argument('-device_id', '--device', type=str, help='Cuda device')
    # parser.add_argument('-f', '--flag', action='store_true', help='A boolean flag')
    args = parser.parse_args()
    main(args.dataset, args.device)

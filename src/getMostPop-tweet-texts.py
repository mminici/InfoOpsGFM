import pandas as pd
import numpy as np
import nltk
import argparse
from nltk.corpus import stopwords
from my_utils import set_seed, setup_env
from tweetSimUtil import msg_clean

CONTROL_USERS_FILENAME = {'UAE_sample': 'control_driver_tweets_uae_082019.jsonl',
                          'cuba': 'control_driver_tweets_cuba_082020.jsonl'}
IO_USERS_FILENAME = {'UAE_sample': 'uae_082019_tweets_csv_unhashed.csv',
                     'cuba': 'cuba_082020_tweets_csv_unhashed.csv'}


def main(seed, dataset_name, MIN_TWEETS, TOP_POPULAR_TWEETS, hyper_params):
    set_seed(seed)
    device, base_dir, interim_data_dir, data_dir = setup_env('0', dataset_name, hyper_params)

    print('Importing Control file...\n',
           base_dir / 'data' / 'raw' / dataset_name / CONTROL_USERS_FILENAME[dataset_name])
    control_df = pd.read_json(base_dir / 'data' / 'raw' / dataset_name / CONTROL_USERS_FILENAME[dataset_name],
                              lines=True)
    print('Importing IO drivers file...\n',
          base_dir / 'data' / 'raw' / dataset_name / IO_USERS_FILENAME[dataset_name])
    iodrivers_df = pd.read_csv(base_dir / 'data' / 'raw' / dataset_name / IO_USERS_FILENAME[dataset_name], sep=",")

    print('Total number of IO drivers: ', iodrivers_df.userid.nunique())

    userStats = iodrivers_df.userid.value_counts()
    userStats = userStats[(userStats >= MIN_TWEETS)]
    print(
        f'There are {len(userStats)} users with at least {MIN_TWEETS} tweets (out of {iodrivers_df.userid.nunique()})')

    my_iodrivers_df = iodrivers_df.copy()
    my_iodrivers_df = my_iodrivers_df[my_iodrivers_df.userid.isin(userStats.index)]

    # Clean tweet text
    # Downloading Stopwords
    nltk.download('stopwords')
    # Load English Stop Words
    stopword = stopwords.words('english')
    # Clean messages and remove short tweets
    my_iodrivers_df['clean_tweet'] = my_iodrivers_df['tweet_text'].astype(str).apply(
        lambda x: msg_clean(x, stopword=stopword))
    my_iodrivers_df = my_iodrivers_df[my_iodrivers_df['clean_tweet'].apply(lambda x: len(x.split(' ')) > 4)]

    # Restrict the tweets for each user to their top-5 popular tweets
    retweet_counts = my_iodrivers_df[['tweetid', 'userid', 'retweet_tweetid']][
        'retweet_tweetid'].value_counts().reset_index()
    retweet_counts.columns = ['tweetid', 'retweet_count']

    missing_tweetids = set(my_iodrivers_df.tweetid.unique()) - set(retweet_counts['tweetid'])
    missing_df = pd.DataFrame({'tweetid': list(missing_tweetids), 'retweet_count': 0})

    retweet_counts = pd.concat([retweet_counts, missing_df], ignore_index=True)
    # Convert the dictionary to a DataFrame
    popularity_df = retweet_counts.copy()
    popularity_df.columns = ['tweetid', 'popularity']
    # Merge the original DataFrame with the popularity DataFrame
    merged_df = pd.merge(my_iodrivers_df, popularity_df, on='tweetid')
    # Group by 'userid' and sort tweets by popularity within each group
    top_tweets = (merged_df.sort_values(['userid', 'popularity'], ascending=[True, False])
                  .groupby('userid')
                  .head(5))

    result_df = top_tweets.drop(columns=['popularity'])
    result_df[['userid', 'clean_tweet']].to_csv(base_dir / 'data' / 'processed' / dataset_name / 'IO_mostPop_tweet_texts.csv')
    # Repeat with control users
    control_df['userid'] = control_df['user'].apply(lambda x: np.int64(x['id']))
    print('Total number of Control users: ', control_df.userid.nunique())
    userStats = control_df.userid.value_counts()
    userStats = userStats[(userStats >= MIN_TWEETS)]
    print(
        f'There are {len(userStats)} CONTROL users with at least {MIN_TWEETS} tweets (out of {control_df.userid.nunique()})')
    my_control_df = control_df.copy()
    my_control_df = my_control_df[my_control_df.userid.isin(userStats.index)]
    # Clean messages and remove short tweets
    my_control_df['clean_tweet'] = my_control_df['full_text'].astype(str).apply(
        lambda x: msg_clean(x, stopword=stopword))
    my_control_df = my_control_df[my_control_df['clean_tweet'].apply(lambda x: len(x.split(' ')) > 4)]
    print('DONE')
    # Remapping column names
    remappingCols = {column_name: column_name for column_name in my_control_df.columns}
    remappingCols['id'] = 'tweetid'
    my_control_df = my_control_df.rename(columns=remappingCols)
    # since for control users we do not have engagement information (eg, likes, RTs)
    # we compute the number of retweets for each tweet from our own dataset
    control_df_count_pop = control_df[['user', 'retweeted_status', 'id']]
    control_df_count_pop.dropna(inplace=True)
    control_df_count_pop['retweet_id'] = control_df_count_pop['retweeted_status'].apply(lambda x: int(dict(x)['id']))
    control_df_count_pop['userid'] = control_df_count_pop['user'].apply(lambda x: int(dict(x)['id']))
    control_df_count_pop = control_df_count_pop[['id', 'userid', 'retweet_id']]
    control_df_count_pop.columns = ['tweetid', 'userid', 'retweet_tweetid']
    # Step 1: Count the number of times each tweetid was retweeted
    retweet_counts = control_df_count_pop['retweet_tweetid'].value_counts().reset_index()
    retweet_counts.columns = ['tweetid', 'retweet_count']
    # Step 2: Ensure all tweetids in the set are present in the DataFrame with a count of 0 if not retweeted
    missing_tweetids = set(my_control_df.tweetid.unique()) - set(retweet_counts['tweetid'])
    missing_df = pd.DataFrame({'tweetid': list(missing_tweetids), 'retweet_count': 0})
    # Concatenate the dataframes
    retweet_counts = pd.concat([retweet_counts, missing_df], ignore_index=True)
    # Convert the dictionary to a DataFrame
    popularity_df = retweet_counts.copy()
    popularity_df.columns = ['tweetid', 'popularity']
    # Merge the original DataFrame with the popularity DataFrame
    merged_df = pd.merge(my_control_df, popularity_df, on='tweetid')
    # Group by 'userid' and sort tweets by popularity within each group
    top_tweets = (merged_df.sort_values(['userid', 'popularity'], ascending=[True, False])
                  .groupby('userid')
                  .head(TOP_POPULAR_TWEETS))
    # Drop the 'popularity' column to match the original DataFrame's structure
    result_df = top_tweets.drop(columns=['popularity'])
    result_df[['userid', 'clean_tweet']].to_csv(
        base_dir / 'data' / 'processed' / dataset_name / 'CONTROL_mostPop_tweet_texts.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess dataset to get all similarity networks")
    parser.add_argument('-dataset_name', '--dataset', type=str, default='cuba')
    parser.add_argument('-seed', '--seed', type=int, default=12121995)
    parser.add_argument('-min_tweets', '--min_tweets', type=int, default=5)
    parser.add_argument('-top_popular_tweets', '--top_pop', type=int, default=5)
    # parser.add_argument('-f', '--flag', action='store_true', help='A boolean flag')
    args = parser.parse_args()
    hyper_parameters = {'train_perc': .6, 'val_perc': .2, 'test_perc': .2,
                        'aggr_type': 'mean', 'num_splits': 5, 'seed': args.seed,
                        'tsim_th': .99}
    main(args.seed, args.dataset, args.min_tweets, args.top_pop, hyper_parameters)

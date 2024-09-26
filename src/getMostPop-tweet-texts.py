import pandas as pd
import numpy as np
import nltk
import argparse
import my_utils
from nltk.corpus import stopwords
from my_utils import set_seed, setup_env
from tweetSimUtil import msg_clean, preprocess_text

CONTROL_USERS_FILENAME = {'UAE_sample': 'control_driver_tweets_uae_082019.jsonl',
                          'cuba': 'control_driver_tweets_cuba_082020.jsonl'}
IO_USERS_FILENAME = {'UAE_sample': 'uae_082019_tweets_csv_unhashed.csv',
                     'cuba': 'cuba_082020_tweets_csv_unhashed.csv'}


def main_by_lang(seed, dataset_name, MIN_TWEETS, TOP_POPULAR_TWEETS, hyper_params, lang):
    set_seed(seed)
    device, base_dir, interim_data_dir, data_dir = setup_env('0', dataset_name, hyper_params)

    print(base_dir / 'data' / 'raw' / dataset_name / f'{dataset_name}_tweets_control.pkl.gz')
    control_df = my_utils.read_compressed_pickle(
        base_dir / 'data' / 'raw' / dataset_name / f'{dataset_name}_tweets_control.pkl.gz')
    control_df['userid'] = control_df['userid'].apply(lambda x: np.int64(x))
    print('Importing IO drivers file...')
    print(base_dir / 'data' / 'raw' / dataset_name / f'{dataset_name}_tweets_io.pkl.gz')
    iodrivers_df = my_utils.read_compressed_pickle(
        base_dir / 'data' / 'raw' / dataset_name / f'{dataset_name}_tweets_io.pkl.gz')

    def convert_to_int(x):
        try:
            toreturn = np.int64(x)
        except ValueError:
            print(f'{x} is not an integer')
            return None
        return toreturn

    iodrivers_df['userid'] = iodrivers_df['userid'].apply(lambda x: convert_to_int(x))
    iodrivers_df = iodrivers_df[~iodrivers_df.userid.isna()]

    print('Total number of IO drivers: ', iodrivers_df.userid.nunique())

    userStats = iodrivers_df.userid.value_counts()
    userStats = userStats[(userStats >= MIN_TWEETS)]
    print(f'There are IO {len(userStats)} users with at least {MIN_TWEETS} tweets (out of {iodrivers_df.userid.nunique()})')

    my_iodrivers_df = iodrivers_df.copy()
    my_iodrivers_df = my_iodrivers_df[my_iodrivers_df.userid.isin(userStats.index)]

    # Preprocess texts
    my_iodrivers_df = preprocess_text(my_iodrivers_df, 'tweet_text')

    # Clean tweet text
    nltk.download('stopwords')
    stopword = stopwords.words('english')
    my_iodrivers_df['clean_tweet'] = my_iodrivers_df['tweet_text'].astype(str).apply(
        lambda x: msg_clean(x, stopword=stopword))
    my_iodrivers_df = my_iodrivers_df[my_iodrivers_df['clean_tweet'].apply(lambda x: len(x.split(' ')) > 4)]

    # Compute tweet popularity across all languages
    retweet_counts = my_iodrivers_df[['tweetid', 'userid', 'retweet_tweetid']]['retweet_tweetid'].value_counts().reset_index()
    retweet_counts.columns = ['tweetid', 'retweet_count']

    missing_tweetids = set(my_iodrivers_df.tweetid.unique()) - set(retweet_counts['tweetid'])
    missing_df = pd.DataFrame({'tweetid': list(missing_tweetids), 'retweet_count': 0})

    retweet_counts = pd.concat([retweet_counts, missing_df], ignore_index=True)
    popularity_df = retweet_counts.copy()
    popularity_df.columns = ['tweetid', 'popularity']

    # Merge the original DataFrame with the popularity DataFrame
    merged_df = pd.merge(my_iodrivers_df, popularity_df, on='tweetid')

    # Filter to only keep the most popular tweets in the specified language
    lang_df = merged_df[merged_df['tweet_language'] == lang]

    # Group by 'userid' and sort tweets by popularity within each group
    top_tweets = (lang_df.sort_values(['userid', 'popularity'], ascending=[True, False])
                  .groupby('userid')
                  .head(TOP_POPULAR_TWEETS))

    result_df = top_tweets.drop(columns=['popularity'])
    result_df[['userid', 'clean_tweet']].to_csv(
        base_dir / 'data' / 'processed' / dataset_name / f'IO_mostPop{TOP_POPULAR_TWEETS}_lang{lang}_tweet_texts.csv')

    # Repeat with control users
    userStats = control_df.userid.value_counts()
    userStats = userStats[(userStats >= MIN_TWEETS)]
    print(f'There are Control {len(userStats)} users with at least {MIN_TWEETS} tweets (out of {control_df.userid.nunique()})')

    my_control_df = control_df.copy()
    my_control_df = my_control_df[my_control_df.userid.isin(userStats.index)]

    # Preprocess texts
    my_control_df = preprocess_text(my_control_df, 'tweet_text')

    nltk.download('stopwords')
    stopword = stopwords.words('english')
    my_control_df['clean_tweet'] = my_control_df['tweet_text'].astype(str).apply(
        lambda x: msg_clean(x, stopword=stopword))
    my_control_df = my_control_df[my_control_df['clean_tweet'].apply(lambda x: len(x.split(' ')) > 4)]

    # Compute tweet popularity across all languages
    retweet_counts = my_control_df[['tweetid', 'userid', 'retweet_tweetid']]['retweet_tweetid'].value_counts().reset_index()
    retweet_counts.columns = ['tweetid', 'retweet_count']

    missing_tweetids = set(my_control_df.tweetid.unique()) - set(retweet_counts['tweetid'])
    missing_df = pd.DataFrame({'tweetid': list(missing_tweetids), 'retweet_count': 0})

    retweet_counts = pd.concat([retweet_counts, missing_df], ignore_index=True)
    popularity_df = retweet_counts.copy()
    popularity_df.columns = ['tweetid', 'popularity']

    # Merge the original DataFrame with the popularity DataFrame
    merged_df = pd.merge(my_control_df, popularity_df, on='tweetid')

    # Filter to only keep the most popular tweets in the specified language
    lang_df = merged_df[merged_df['tweet_language'] == lang]

    # Group by 'userid' and sort tweets by popularity within each group
    top_tweets = (lang_df.sort_values(['userid', 'popularity'], ascending=[True, False])
                  .groupby('userid')
                  .head(TOP_POPULAR_TWEETS))

    result_df = top_tweets.drop(columns=['popularity'])
    result_df[['userid', 'clean_tweet']].to_csv(
        base_dir / 'data' / 'processed' / dataset_name / f'CONTROL_mostPop{TOP_POPULAR_TWEETS}_lang{lang}_tweet_texts.csv')


def main(seed, dataset_name, MIN_TWEETS, TOP_POPULAR_TWEETS, hyper_params):
    set_seed(seed)
    device, base_dir, interim_data_dir, data_dir = setup_env('0', dataset_name, hyper_params)

    print(base_dir / 'data' / 'raw' / dataset_name / f'{dataset_name}_tweets_control.pkl.gz')
    control_df = my_utils.read_compressed_pickle(
        base_dir / 'data' / 'raw' / dataset_name / f'{dataset_name}_tweets_control.pkl.gz')
    control_df['userid'] = control_df['userid'].apply(lambda x: np.int64(x))
    print('Importing IO drivers file...')
    print(base_dir / 'data' / 'raw' / dataset_name / f'{dataset_name}_tweets_io.pkl.gz')
    iodrivers_df = my_utils.read_compressed_pickle(
        base_dir / 'data' / 'raw' / dataset_name / f'{dataset_name}_tweets_io.pkl.gz')

    def convert_to_int(x):
        try:
            toreturn = np.int64(x)
        except ValueError:
            print(f'{x} is not an integer')
            return None
        return toreturn

    # iodrivers_df['userid'] = iodrivers_df['userid'].apply(lambda x: np.int64(x))
    iodrivers_df['userid'] = iodrivers_df['userid'].apply(lambda x: convert_to_int(x))
    iodrivers_df = iodrivers_df[~iodrivers_df.userid.isna()]

    print('Total number of IO drivers: ', iodrivers_df.userid.nunique())

    userStats = iodrivers_df.userid.value_counts()
    userStats = userStats[(userStats >= MIN_TWEETS)]
    print(
        f'There are IO {len(userStats)} users with at least {MIN_TWEETS} tweets (out of {iodrivers_df.userid.nunique()})')

    my_iodrivers_df = iodrivers_df.copy()
    my_iodrivers_df = my_iodrivers_df[my_iodrivers_df.userid.isin(userStats.index)]

    # Preprocess texts
    my_iodrivers_df = preprocess_text(my_iodrivers_df, 'tweet_text')

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
                  .head(TOP_POPULAR_TWEETS))

    result_df = top_tweets.drop(columns=['popularity'])
    result_df[['userid', 'clean_tweet']].to_csv(
        base_dir / 'data' / 'processed' / dataset_name / f'IO_mostPop{TOP_POPULAR_TWEETS}_tweet_texts.csv')
    # Repeat with control users
    userStats = control_df.userid.value_counts()
    userStats = userStats[(userStats >= MIN_TWEETS)]
    print(
        f'There are Control {len(userStats)} users with at least {MIN_TWEETS} tweets (out of {control_df.userid.nunique()})')

    my_control_df = control_df.copy()
    my_control_df = my_control_df[my_control_df.userid.isin(userStats.index)]

    # Preprocess texts
    my_control_df = preprocess_text(my_control_df, 'tweet_text')

    # Clean tweet text
    # Downloading Stopwords
    nltk.download('stopwords')
    # Load English Stop Words
    stopword = stopwords.words('english')
    # Clean messages and remove short tweets
    my_control_df['clean_tweet'] = my_control_df['tweet_text'].astype(str).apply(
        lambda x: msg_clean(x, stopword=stopword))
    my_control_df = my_control_df[my_control_df['clean_tweet'].apply(lambda x: len(x.split(' ')) > 4)]

    # Restrict the tweets for each user to their top-5 popular tweets
    retweet_counts = my_control_df[['tweetid', 'userid', 'retweet_tweetid']][
        'retweet_tweetid'].value_counts().reset_index()
    retweet_counts.columns = ['tweetid', 'retweet_count']

    missing_tweetids = set(my_control_df.tweetid.unique()) - set(retweet_counts['tweetid'])
    missing_df = pd.DataFrame({'tweetid': list(missing_tweetids), 'retweet_count': 0})

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

    result_df = top_tweets.drop(columns=['popularity'])
    result_df[['userid', 'clean_tweet']].to_csv(
        base_dir / 'data' / 'processed' / dataset_name / f'CONTROL_mostPop{TOP_POPULAR_TWEETS}_tweet_texts.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess dataset to get all similarity networks")
    parser.add_argument('-dataset_name', '--dataset', type=str, default='cuba')
    parser.add_argument('-seed', '--seed', type=int, default=12121995)
    parser.add_argument('-min_tweets', '--min_tweets', type=int, default=10)
    parser.add_argument('-top_popular_tweets', '--top_pop', type=int, default=5)
    parser.add_argument('-language', '--lang', type=str, default='en')
    # parser.add_argument('-f', '--flag', action='store_true', help='A boolean flag')
    args = parser.parse_args()
    hyper_parameters = {'train_perc': .6, 'val_perc': .2, 'test_perc': .2,
                        'aggr_type': 'mean', 'num_splits': 5, 'seed': args.seed,
                        'tsim_th': .7, 'min_tweets': args.min_tweets}
    # main(args.seed, args.dataset, args.min_tweets, args.top_pop, hyper_parameters)
    main_by_lang(args.seed, args.dataset, args.min_tweets, args.top_pop, hyper_parameters, args.lang)

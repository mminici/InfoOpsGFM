from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import networkx as nx
import numpy as np
from nltk.corpus import stopwords
import nltk
import re
import os
import warnings

warnings.filterwarnings("ignore")
from datetime import datetime
from datetime import timedelta


def get_tweet_timestamp(tid):
    try:
        offset = 1288834974657
        tstamp = (tid >> 22) + offset
        utcdttime = datetime.utcfromtimestamp(tstamp / 1000)
        return utcdttime
    except:
        return None


def process_data(tweet_df):
    tweet_df['quoted_tweet_tweetid'] = tweet_df['quoted_tweet_tweetid'].astype('Int64')
    tweet_df['retweet_tweetid'] = tweet_df['retweet_tweetid'].astype('Int64')

    # Tweet type classification
    tweet_type = []
    for i in range(tweet_df.shape[0]):
        if pd.notnull(tweet_df['quoted_tweet_tweetid'].iloc[i]):
            if pd.notnull(tweet_df['retweet_tweetid'].iloc[i]):
                if pd.notnull(tweet_df['in_reply_to_tweetid'].iloc[i]):
                    continue
                else:
                    tweet_type.append('retweet')
            else:
                if pd.notnull(tweet_df['in_reply_to_tweetid'].iloc[i]):
                    tweet_type.append('reply')
                else:
                    tweet_type.append('quoted')
        else:
            if pd.notnull(tweet_df['retweet_tweetid'].iloc[i]):
                if pd.notnull(tweet_df['in_reply_to_tweetid'].iloc[i]):
                    continue
                else:
                    tweet_type.append('retweet')
            else:
                if pd.notnull(tweet_df['in_reply_to_tweetid'].iloc[i]):
                    tweet_type.append('reply')
                else:
                    tweet_type.append('original')
    tweet_df['tweet_type'] = tweet_type
    tweet_df = tweet_df[tweet_df.tweet_type != 'retweet']

    return tweet_df


def preprocess_text(df):
    # Cleaning tweets in en language
    # Removing RT Word from Messages
    df['tweet_text'] = df['tweet_text'].str.lstrip('RT')
    # Removing selected punctuation marks from Messages
    df['tweet_text'] = df['tweet_text'].str.replace(":", '')
    df['tweet_text'] = df['tweet_text'].str.replace(";", '')
    df['tweet_text'] = df['tweet_text'].str.replace(".", '')
    df['tweet_text'] = df['tweet_text'].str.replace(",", '')
    df['tweet_text'] = df['tweet_text'].str.replace("!", '')
    df['tweet_text'] = df['tweet_text'].str.replace("&", '')
    df['tweet_text'] = df['tweet_text'].str.replace("-", '')
    df['tweet_text'] = df['tweet_text'].str.replace("_", '')
    df['tweet_text'] = df['tweet_text'].str.replace("$", '')
    df['tweet_text'] = df['tweet_text'].str.replace("/", '')
    df['tweet_text'] = df['tweet_text'].str.replace("?", '')
    df['tweet_text'] = df['tweet_text'].str.replace("''", '')
    # Lowercase
    df['tweet_text'] = df['tweet_text'].str.lower()

    return df


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


# Message Clean Function
def msg_clean(msg, stopword):
    # Remove URL
    msg = re.sub(r'https?://\S+|www\.\S+', " ", msg)

    # Remove Mentions
    msg = re.sub(r'@\w+', ' ', msg)

    # Remove Digits
    msg = re.sub(r'\d+', ' ', msg)

    # Remove HTML tags
    msg = re.sub('r<.*?>', ' ', msg)

    # Remove HTML tags
    msg = re.sub('r<.*?>', ' ', msg)

    # Remove Emoji from text
    msg = remove_emoji(msg)

    # Remove Stop Words
    msg = msg.split()

    msg = " ".join([word for word in msg if word not in stopword])

    return msg


def create_sim_score_df(lims, D, I, search_query1, combined_tweets_df):
    source_idx = []
    target_idx = []
    sim_score = []

    for i in range(len(search_query1)):
        idx = I[lims[i]:lims[i + 1]]
        sim = D[lims[i]:lims[i + 1]]
        for j in range(len(idx)):
            source_idx.append(i)
            target_idx.append(idx[j])
            sim_score.append(sim[j])

    sim_score_df = pd.DataFrame(list(zip(source_idx, target_idx, sim_score)),
                                columns=['source_idx', 'target_idx', 'sim_score'])
    del source_idx
    del target_idx
    del sim_score
    sim_score_df = sim_score_df.query("source_idx != target_idx")
    sim_score_df['combined_idx'] = sim_score_df[['source_idx', 'target_idx']].apply(tuple, axis=1)
    sim_score_df['combined_idx'] = sim_score_df['combined_idx'].apply(sorted)
    sim_score_df['combined_idx'] = sim_score_df['combined_idx'].transform(lambda k: tuple(k))
    sim_score_df = sim_score_df.drop_duplicates(subset=['combined_idx'], keep='first')
    sim_score_df.reset_index(inplace=True)
    sim_score_df = sim_score_df.loc[:, ~sim_score_df.columns.str.contains('index')]
    sim_score_df.drop(['combined_idx'], inplace=True, axis=1)

    df_join = pd.merge(pd.merge(sim_score_df, combined_tweets_df, left_on='source_idx', right_on='my_idx', how='inner'),
                       combined_tweets_df, left_on='target_idx', right_on='my_idx', how='inner')

    result = df_join[['userid_x', 'userid_y', 'clean_tweet_x', 'clean_tweet_y', 'sim_score']]
    result = result.rename(columns={'userid_x': 'source_user',
                                    'userid_y': 'target_user',
                                    'clean_tweet_x': 'source_text',
                                    'clean_tweet_y': 'target_text'})
    return result


# MAIN FUNCTION
# Data assumptions:
#   - datasetsPaths: list containing the absolute paths referring to the datasets to analyze
#   - outputDir: directory where to save temporary files
# To solve computational issues, the function will create multiple output files of users sharing
# similar texts that will need to then be merged into a network using the getSimilarityNetwork function (see below)

def textSim(control, treated, outputDir, stopword, cudaId='1'):
    os.environ["CUDA_VISIBLE_DEVICES"] = cudaId
    control = control[['id', 'full_text', 'lang', 'user', 'created_at']]
    control.columns = ['tweetid', 'tweet_text', 'lang', 'user', 'created_at']
    treated = treated[['tweetid', 'userid', 'tweet_time', 'tweet_language', 'tweet_text']]

    print('\t Preprocessing treated tweet text...')
    pos_en_df_all = preprocess_text(treated)
    del treated
    print('\t Preprocessing control tweet text...')
    neg_en_df_all = preprocess_text(control)
    del control

    pos_en_df_all['tweet_text'] = pos_en_df_all['tweet_text'].replace(',', '')
    neg_en_df_all['tweet_text'] = neg_en_df_all['tweet_text'].replace(',', '')

    pos_en_df_all['clean_tweet'] = pos_en_df_all['tweet_text'].astype(str).apply(
        lambda x: msg_clean(x, stopword=stopword))
    neg_en_df_all['clean_tweet'] = neg_en_df_all['tweet_text'].astype(str).apply(
        lambda x: msg_clean(x, stopword=stopword))

    pos_en_df_all = pos_en_df_all[pos_en_df_all['clean_tweet'].apply(lambda x: len(x.split(' ')) > 4)]
    neg_en_df_all = neg_en_df_all[neg_en_df_all['clean_tweet'].apply(lambda x: len(x.split(' ')) > 4)]

    pos_en_df_all['tweet_time'] = pos_en_df_all['tweetid'].apply(lambda x: get_tweet_timestamp(x))
    neg_en_df_all['tweet_time'] = neg_en_df_all['tweetid'].apply(lambda x: get_tweet_timestamp(x))

    date = pos_en_df_all['tweet_time'].min().date()
    finalDate = pos_en_df_all['tweet_time'].max().date()

    i = 1

    print('\t Embed tweets in sliding windows')
    while date <= finalDate:
        pos_en_df = pos_en_df_all.loc[(pos_en_df_all['tweet_time'].dt.date >= date) & (
                pos_en_df_all['tweet_time'].dt.date < date + timedelta(days=1))]
        neg_en_df = neg_en_df_all.loc[(neg_en_df_all['tweet_time'].dt.date >= date) & (
                neg_en_df_all['tweet_time'].dt.date < date + timedelta(days=1))]

        combined_tweets_df = pd.concat([pos_en_df, neg_en_df], axis=0)
        combined_tweets_df.reset_index(inplace=True)
        combined_tweets_df = combined_tweets_df.loc[:, ~combined_tweets_df.columns.str.contains('index')]

        del pos_en_df
        del neg_en_df

        combined_tweets_df.reset_index(inplace=True)
        combined_tweets_df = combined_tweets_df.rename(columns={'index': 'my_idx'})

        sentences = combined_tweets_df.clean_tweet.tolist()
        encoder = SentenceTransformer('stsb-xlm-r-multilingual')
        plot_embeddings = encoder.encode(sentences)

        try:
            dim1 = plot_embeddings.shape[0]
            dim2 = plot_embeddings.shape[1]  # vector dimension
            assert dim1 > 100
        except:
            date = date + timedelta(days=1)
            print('need to stretch the sliding window')
            continue

        db_vectors1 = plot_embeddings.copy().astype(np.float32)
        a = [i for i in range(plot_embeddings.shape[0])]
        db_ids1 = np.array(a, dtype=np.int64)

        print('\t FAISS utility to normalize vectors')
        faiss.normalize_L2(db_vectors1)

        print('\t FAISS utility to create index')
        index1 = faiss.IndexFlatIP(dim2)
        index1 = faiss.IndexIDMap(index1)  # mapping df index as id
        print('\t FAISS utility to populate DB')
        index1.add_with_ids(db_vectors1, db_ids1)

        search_query1 = plot_embeddings.copy().astype(np.float32)

        faiss.normalize_L2(search_query1)

        init_threshold = 0.7
        print('\t FAISS utility to query text pairs more similar than 0.7 cosine sim')
        lims, D, I = index1.range_search(x=search_query1, thresh=init_threshold)

        print('Retrieved results of index search')
        sim_score_df = create_sim_score_df(lims, D, I, search_query1, combined_tweets_df)

        print('Generated Similarity Score DataFrame')

        del combined_tweets_df

        for threshold in np.arange(0.7, 1.01, 0.05):
            print("Threshold: ", threshold)

            sim_score_temp_df = sim_score_df[sim_score_df.sim_score >= threshold]

            text_sim_network = sim_score_temp_df[['source_user', 'target_user']]
            text_sim_network = text_sim_network.drop_duplicates(subset=['source_user', 'target_user'], keep='first')

            outputfile = outputDir / f'threshold_{threshold}_{i}.csv'
            text_sim_network.to_csv(outputfile)

        date = date + timedelta(days=1)
        i += 1


# to run after the textSim function inputDir: path of the directory containing the similarity files; it corresponds
# to the outputDir used in the textSim function
def getSimilarityNetwork(outputDir):
    files = [f for f in outputDir.iterdir()]
    files.sort()

    d = {'threshold_1.00': [],
         'threshold_0.90': [],
         'threshold_0.95': [],
         'threshold_0.85': [],
         'threshold_0.8': [],
         'threshold_0.75': [],
         'threshold_0.7': []}

    for f in files:
        if f.name[:9] == 'threshold':
            d['_'.join(f.name[:-4].split('_')[:2])[:14]].append(f)

    d_keys = list(d.keys())
    fil = d_keys[0]
    thr = float(fil.split('_')[-1][:4])
    l = d[fil]
    combined = pd.read_csv(l[0])  # (path + l[0])
    combined['weight'] = thr
    for o in l[1:]:
        temp = pd.read_csv(o)
        temp['weight'] = thr
        combined = pd.concat([combined, temp])
    for fil in d_keys[1:]:
        thr = float(fil.split('_')[-1][:4])
        l = d[fil]
        for o in l:
            temp = pd.read_csv(o)
            temp['weight'] = thr
            combined = pd.concat([combined, temp])

    combined.sort_values(by='weight', ascending=False, inplace=True)
    combined.drop_duplicates(subset=['source_user', 'target_user'], inplace=True)
    G = nx.from_pandas_edgelist(combined, source='source_user', target='target_user', edge_attr=['weight'])

    return G


def getTweetSimNetwork(control, treated, outputDir, cudaId='1'):
    # Downloading Stopwords
    nltk.download('stopwords')
    # Load English Stop Words
    stopword = stopwords.words('english')
    outputDir.mkdir(parents=True, exist_ok=True)
    # compute tweet similarity
    textSim(control=control, treated=treated, outputDir=outputDir, stopword=stopword, cudaId=cudaId)
    # get similarity network
    G = getSimilarityNetwork(outputDir)
    print(f'Number of nodes: {G.number_of_nodes()}')
    print(f'Number of edges: {G.number_of_edges()}')
    return G

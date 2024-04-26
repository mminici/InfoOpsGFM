import datetime
import pandas as pd
import numpy as np
import networkx as nx
from pandas import CategoricalDtype
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity


def get_tweet_timestamp(tid):
    try:
        offset = 1288834974657
        tstamp = (tid >> 22) + offset
        utcdttime = datetime.fromtimestamp(tstamp / 1000)
        return utcdttime
    except:
        return None


# Data assumptions: - 2 Pandas dataframes - control: control dataset -> includes only columns ['retweeted_status',
# 'user', 'in_reply_to_status_id', 'full_text', 'id'] - treated: information Operation dataset -> includes only
# columns ['retweet_tweetid', 'userid', 'in_reply_to_tweetid', 'quoted_tweet_tweetid', 'tweet_text', 'tweetid']
# minHashtags: minimum number of hashtags inside an hashtag sequence

def hashSeq(control, treated, minHashtags=5):
    control.replace(np.NaN, None, inplace=True)

    control['engagementParentId'] = control['in_reply_to_status_id']

    retweet_id = []
    names = []
    eng = []
    for row in control[['retweeted_status', 'user', 'in_reply_to_status_id']].values:
        if row[0] != None:
            u = dict(row[0])
            retweet_id.append(u['id'])
            eng.append('retweet')
        elif row[2] != None:
            retweet_id.append(row[2])
            eng.append('reply')
        else:
            retweet_id.append(None)
            eng.append('tweet')
        u = dict(row[1])
        names.append(u['id'])

    control['twitterAuthorScreenname'] = names
    control['retweet_ordinalId'] = retweet_id
    control['engagementType'] = eng
    control['engagementParentId'].fillna(control['retweet_ordinalId'], inplace=True)

    control_filt = control[['twitterAuthorScreenname', 'engagementType', 'engagementParentId']]
    control_filt['contentText'] = control['full_text']
    control_filt['tweetId'] = control['id'].astype(int)
    control_filt['tweet_timestamp'] = control_filt['tweetId'].apply(lambda x: get_tweet_timestamp(x))

    del control

    treated.replace(np.NaN, None, inplace=True)

    retweet_id = []
    names = []
    eng = []
    for row in treated[['retweet_tweetid', 'userid', 'in_reply_to_tweetid', 'quoted_tweet_tweetid']].values:
        if row[0] != None:
            retweet_id.append(row[0])
            eng.append('retweet')
        elif row[2] != None:
            retweet_id.append(row[2])
            eng.append('reply')
        elif row[3] != None:
            retweet_id.append(row[3])
            eng.append('quote tweet')
        else:
            retweet_id.append(None)
            eng.append('tweet')
        names.append(row[1])

    treated['twitterAuthorScreenname'] = names
    treated['engagementType'] = eng
    treated['engagementParentId'] = retweet_id

    treated_filt = treated[['twitterAuthorScreenname', 'engagementType', 'engagementParentId']]
    treated_filt['contentText'] = treated['tweet_text']
    treated_filt['tweetId'] = treated['tweetid'].astype(int)
    treated_filt['tweet_timestamp'] = treated_filt['tweetId'].apply(lambda x: get_tweet_timestamp(x))

    del treated

    cum = pd.concat([control_filt, treated_filt])

    del control_filt, treated_filt

    cum = cum.loc[cum['engagementType'] != 'retweet']
    cum['hashtag_seq'] = ['__'.join([tag.strip("#") for tag in tweet.split() if tag.startswith("#")]) for tweet in
                          cum['contentText'].values.astype(str)]
    cum.drop('contentText', axis=1, inplace=True)
    cum = cum[['twitterAuthorScreenname', 'hashtag_seq']].loc[
        cum['hashtag_seq'].apply(lambda x: len(x.split('__'))) >= i]

    cum.drop_duplicates(inplace=True)

    temp = cum.groupby('hashtag_seq', as_index=False).count()
    cum = cum.loc[cum['hashtag_seq'].isin(temp.loc[temp['twitterAuthorScreenname'] > 1]['hashtag_seq'].to_list())]

    cum['value'] = 1

    hashs = dict(zip(list(cum.hashtag_seq.unique()), list(range(cum.hashtag_seq.unique().shape[0]))))
    cum['hashtag_seq'] = cum['hashtag_seq'].apply(lambda x: hashs[x]).astype(int)
    del hashs

    userid = dict(zip(list(cum.twitterAuthorScreenname.astype(str).unique()),
                      list(range(cum.twitterAuthorScreenname.unique().shape[0]))))
    cum['twitterAuthorScreenname'] = cum['twitterAuthorScreenname'].astype(str).apply(lambda x: userid[x]).astype(int)

    person_c = CategoricalDtype(sorted(cum.twitterAuthorScreenname.unique()), ordered=True)
    thing_c = CategoricalDtype(sorted(cum.hashtag_seq.unique()), ordered=True)

    row = cum.twitterAuthorScreenname.astype(person_c).cat.codes
    col = cum.hashtag_seq.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((cum["value"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
    del row, col, person_c, thing_c

    # cum = pd.pivot_table(cum,'value', 'userid', 'urls', aggfunc='max')
    # cum.fillna(0, inplace = True)

    vectorizer = TfidfTransformer()
    tfidf_matrix = vectorizer.fit_transform(sparse_matrix)
    similarities = cosine_similarity(tfidf_matrix, dense_output=False)

    df_adj = pd.DataFrame(similarities.toarray())
    del similarities
    df_adj.index = userid.keys()
    df_adj.columns = userid.keys()
    G = nx.from_pandas_adjacency(df_adj)
    del df_adj

    G.remove_nodes_from(list(nx.isolates(G)))

    return G

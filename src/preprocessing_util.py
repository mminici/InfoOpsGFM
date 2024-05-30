import pandas as pd
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

from pandas import CategoricalDtype
from scipy.sparse import csr_matrix

from datetime import datetime


def print_network_stats(network):
    print(f'Number of nodes: {network.number_of_nodes()}')
    print(f'Number of edges: {network.number_of_edges()}')


# retrieves tweet's timestamp from its ID
def get_tweet_timestamp(tid):
    try:
        offset = 1288834974657
        tstamp = (tid >> 22) + offset
        utcdttime = datetime.utcfromtimestamp(tstamp / 1000)
        return utcdttime
    except:
        return None


def coRetweet(control, treated):
    # control dataset -> includes only columns ['user', 'retweeted_status', 'id']
    # information Operation dataset -> includes only columns ['tweetid', 'userid', 'retweet_tweetid']
    control = control[['user', 'retweeted_status', 'id']]
    treated = treated[['tweetid', 'userid', 'retweet_tweetid']]

    print('Start preprocessing...')
    print('\t Preprocessing control data...')
    control.dropna(inplace=True)
    treated.dropna(inplace=True)

    control['retweet_id'] = control['retweeted_status'].apply(lambda x: int(dict(x)['id']))
    control['userid'] = control['user'].apply(lambda x: int(dict(x)['id']))
    control = control[['id', 'userid', 'retweet_id']]
    control.columns = ['tweetid', 'userid', 'retweet_tweetid']

    print('\t Preprocessing treated data...')
    treated['retweet_tweetid'] = treated['retweet_tweetid'].astype(int)

    print('\t Putting together control and treated data')
    cum = pd.concat([treated, control])
    filt = cum[['userid', 'tweetid']].groupby(['userid'], as_index=False).count()
    filt = list(filt.loc[filt['tweetid'] >= 20]['userid'])
    cum = cum.loc[cum['userid'].isin(filt)]
    cum = cum[['userid', 'retweet_tweetid']].drop_duplicates()

    temp = cum.groupby('retweet_tweetid', as_index=False).count()
    cum = cum.loc[cum['retweet_tweetid'].isin(temp.loc[temp['userid'] > 1]['retweet_tweetid'].to_list())]

    cum['value'] = 1

    ids = dict(zip(list(cum.retweet_tweetid.unique()), list(range(cum.retweet_tweetid.unique().shape[0]))))
    cum['retweet_tweetid'] = cum['retweet_tweetid'].apply(lambda x: ids[x]).astype(int)
    del ids

    userid = dict(zip(list(cum.userid.astype(str).unique()), list(range(cum.userid.unique().shape[0]))))
    cum['userid'] = cum['userid'].astype(str).apply(lambda x: userid[x]).astype(int)

    person_c = CategoricalDtype(sorted(cum.userid.unique()), ordered=True)
    thing_c = CategoricalDtype(sorted(cum.retweet_tweetid.unique()), ordered=True)

    row = cum.userid.astype(person_c).cat.codes
    col = cum.retweet_tweetid.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((cum["value"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
    del row, col, person_c, thing_c

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
    print_network_stats(G)
    return G


def coURL(control, treated):
    # control dataset -> includes only columns ['user', 'entities', 'id']
    # information Operation dataset -> includes only columns ['tweetid', 'userid', 'urls']
    control = control[['user', 'entities', 'id']]
    treated = treated[['tweetid', 'userid', 'urls']]

    print('Start preprocessing...')
    print('\t Preprocessing control data...')

    control.dropna(inplace=True)

    control['userid'] = control['user'].apply(lambda x: dict(x)['id'])
    control['urls'] = control['entities'].apply(lambda x: dict(x)['urls'])
    control = control[['userid', 'urls']].explode('urls')
    control.dropna(inplace=True)
    control['urls'] = control['urls'].apply(
        lambda x: str(dict(x)['expanded_url'].replace(',', '.')) if x else np.NaN)

    print('\t Preprocessing treated data...')
    treated['urls'] = treated['urls'].astype(str).replace('[]', '').apply(
        lambda x: x[1:-1].replace("'", '').split(',') if len(x) != 0 else '')
    treated = treated.loc[treated['urls'] != ''].explode('urls')

    print('\t Putting together control and treated data')
    cum = pd.concat([control, treated])[['userid', 'urls']].dropna()
    cum.drop_duplicates(inplace=True)

    temp = cum.groupby('urls', as_index=False).count()
    cum = cum.loc[cum['urls'].isin(temp.loc[temp['userid'] > 1]['urls'].to_list())]

    cum['value'] = 1
    urls = dict(zip(list(cum.urls.unique()), list(range(cum.urls.unique().shape[0]))))
    cum['urls'] = cum['urls'].apply(lambda x: urls[x]).astype(int)
    del urls

    userid = dict(zip(list(cum.userid.astype(str).unique()), list(range(cum.userid.unique().shape[0]))))
    cum['userid'] = cum['userid'].astype(str).apply(lambda x: userid[x]).astype(int)

    person_c = CategoricalDtype(sorted(cum.userid.unique()), ordered=True)
    thing_c = CategoricalDtype(sorted(cum.urls.unique()), ordered=True)

    row = cum.userid.astype(person_c).cat.codes
    col = cum.urls.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((cum["value"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
    del row, col, person_c, thing_c

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
    print_network_stats(G)
    return G


def hashSeq(control, treated, minHashtags=5):
    print('Start preprocessing...')
    print('\t Preprocessing control data...')
    # control dataset -> includes only columns ['retweeted_status', 'user', 'in_reply_to_status_id', 'full_text', 'id']
    control = control[['retweeted_status', 'user', 'in_reply_to_status_id', 'full_text', 'id']]
    # information Operation dataset -> includes only
    # columns ['retweet_tweetid', 'userid', 'in_reply_to_tweetid', 'quoted_tweet_tweetid', 'tweet_text', 'tweetid']
    treated = treated[
        ['retweet_tweetid', 'userid', 'in_reply_to_tweetid', 'quoted_tweet_tweetid', 'tweet_text', 'tweetid']]
    # Start preprocessing
    control.replace(np.NaN, None, inplace=True)
    control['engagementParentId'] = control['in_reply_to_status_id']
    print('\t For-Loop')
    retweet_id = []
    names = []
    eng = []
    for row in control[['retweeted_status', 'user', 'in_reply_to_status_id']].values:
        if row[0] is not None:
            u = dict(row[0])
            retweet_id.append(u['id'])
            eng.append('retweet')
        elif row[2] is not None:
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

    print('\t Preprocessing treated data...')
    treated.replace(np.NaN, None, inplace=True)

    retweet_id = []
    names = []
    eng = []
    print('\t For-Loop')
    for row in treated[['retweet_tweetid', 'userid', 'in_reply_to_tweetid', 'quoted_tweet_tweetid']].values:
        if row[0] is not None:
            retweet_id.append(row[0])
            eng.append('retweet')
        elif row[2] is not None:
            retweet_id.append(row[2])
            eng.append('reply')
        elif row[3] is not None:
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

    print('\t Putting together control and treated data')
    cum = pd.concat([control_filt, treated_filt])

    del control_filt, treated_filt

    print('\t Filter out some entries')
    cum = cum.loc[cum['engagementType'] != 'retweet']
    cum['hashtag_seq'] = ['__'.join([tag.strip("#") for tag in tweet.split() if tag.startswith("#")]) for tweet in
                          cum['contentText'].values.astype(str)]
    cum.drop('contentText', axis=1, inplace=True)
    cum = cum[['twitterAuthorScreenname', 'hashtag_seq']].loc[
        cum['hashtag_seq'].apply(lambda x: len(x.split('__'))) >= minHashtags]

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

    print('\t Computing TF-IDF matrix')
    vectorizer = TfidfTransformer()
    tfidf_matrix = vectorizer.fit_transform(sparse_matrix)
    print('\t Computing cosine similarity')
    similarities = cosine_similarity(tfidf_matrix, dense_output=False)

    print('\t Computing graph representation')
    df_adj = pd.DataFrame(similarities.toarray())
    del similarities
    df_adj.index = userid.keys()
    df_adj.columns = userid.keys()
    G = nx.from_pandas_adjacency(df_adj)
    del df_adj

    G.remove_nodes_from(list(nx.isolates(G)))
    print_network_stats(G)
    return G


def fastRetweet(control, treated, timeInterval=10):
    # control dataset -> includes only columns ['user', 'retweeted_status', 'id']
    # information Operation dataset -> includes only columns ['tweetid', 'userid', 'retweet_tweetid', 'retweet_userid']
    control = control[['user', 'retweeted_status', 'id']]
    treated = treated[['tweetid', 'userid', 'retweet_tweetid', 'retweet_userid']]

    print('Start preprocessing...')
    print('\t Preprocessing control data...')
    control.dropna(inplace=True)
    treated.dropna(inplace=True)

    control['retweet_id'] = control['retweeted_status'].apply(lambda x: int(dict(x)['id']))
    control['retweet_userid'] = control['retweeted_status'].apply(lambda x: int(dict(dict(x)['user'])['id']))
    control['userid'] = control['user'].apply(lambda x: int(dict(x)['id']))
    control['tweet_timestamp'] = control['id'].apply(lambda x: get_tweet_timestamp(int(x)))
    control['retweet_timestamp'] = control['retweet_id'].apply(lambda x: get_tweet_timestamp(int(x)))
    control = control[['id', 'userid', 'retweet_id', 'tweet_timestamp', 'retweet_timestamp', 'retweet_userid']]
    control.columns = ['tweetid', 'userid', 'retweet_tweetid', 'tweet_timestamp', 'retweet_timestamp',
                       'retweet_userid']

    print('\t Preprocessing treated data...')
    treated['retweet_tweetid'] = treated['retweet_tweetid'].astype(int)
    treated['tweet_timestamp'] = treated['tweetid'].apply(lambda x: get_tweet_timestamp(int(x)))
    treated['retweet_timestamp'] = treated['retweet_tweetid'].apply(lambda x: get_tweet_timestamp(int(x)))

    print('\t Computing Retweet Delta time...')
    treated['delta'] = (treated['tweet_timestamp'] - treated['retweet_timestamp']).dt.seconds
    control['delta'] = (control['tweet_timestamp'] - control['retweet_timestamp']).dt.seconds

    print('\t Putting together control and treated data')
    cumulative = pd.concat(
        [treated[['userid', 'retweet_userid', 'delta']], control[['userid', 'retweet_userid', 'delta']]])
    cumulative['userid'].astype(int).astype(str)
    cumulative = cumulative.loc[cumulative['delta'] <= timeInterval]

    cumulative = cumulative.groupby(['userid', 'retweet_userid'], as_index=False).count()

    cumulative = cumulative.loc[cumulative['delta'] > 1]

    # Note: this line creates a networkX graph, but there are still pandas operations after...
    # cum = nx.from_pandas_edgelist(cumulative, 'userid', 'retweet_userid','delta')
    # cum = cum.loc[cum['delta'] > 1]

    urls = dict(zip(list(cumulative.retweet_userid.unique()), list(range(cumulative.retweet_userid.unique().shape[0]))))
    cumulative['retweet_userid'] = cumulative['retweet_userid'].apply(lambda x: urls[x]).astype(int)
    del urls

    userid = dict(zip(list(cumulative.userid.astype(str).unique()), list(range(cumulative.userid.unique().shape[0]))))
    cumulative['userid'] = cumulative['userid'].astype(str).apply(lambda x: userid[x]).astype(int)

    person_c = CategoricalDtype(sorted(cumulative.userid.unique()), ordered=True)
    thing_c = CategoricalDtype(sorted(cumulative.retweet_userid.unique()), ordered=True)

    row = cumulative.userid.astype(person_c).cat.codes
    col = cumulative.retweet_userid.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((cumulative["delta"], (row, col)),
                               shape=(person_c.categories.size, thing_c.categories.size))
    del row, col, person_c, thing_c

    vectorizer = TfidfTransformer()
    tfidf_matrix = vectorizer.fit_transform(sparse_matrix)
    similarities = cosine_similarity(tfidf_matrix, dense_output=False)

    df_adj = pd.DataFrame(similarities.toarray())
    del similarities
    df_adj.index = userid.keys()
    df_adj.columns = userid.keys()
    G = nx.from_pandas_adjacency(df_adj)
    del df_adj

    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    print_network_stats(G)
    return G

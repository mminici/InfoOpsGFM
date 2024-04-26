import pandas as pd
import networkx as nx

# retrieves tweet's timestamp from its ID
from pandas import CategoricalDtype
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime


def get_tweet_timestamp(tid):
    try:
        offset = 1288834974657
        tstamp = (tid >> 22) + offset
        utcdttime = datetime.fromtimestamp(tstamp / 1000)
        return utcdttime
    except:
        return None


# Data assumptions:
#   - 2 Pandas dataframes
#     - control: control dataset -> includes only columns ['user', 'retweeted_status', 'id']
#     - treated: information Operation dataset -> includes only columns ['tweetid', 'userid', 'retweet_tweetid', 'retweet_userid']
# timeInterval: time distance in seconds between retweet and original tweet under which a retweet is considered fast

def fastRetweet(control, treated, timeInterval=10):
    control.dropna(inplace=True)
    treated.dropna(inplace=True)

    control['retweet_id'] = control['retweeted_status'].apply(lambda x: int(dict(x)['id']))
    control['retweet_userid'] = control['retweeted_status'].apply(lambda x: int(dict(dict(x)['user'])['id']))
    control['userid'] = control['user'].apply(lambda x: int(dict(x)['id']))
    control['tweet_timestamp'] = control['id'].apply(lambda x: get_tweet_timestamp(int(x)))
    control['retweet_timestamp'] = control['retweet_id'].apply(lambda x: get_tweet_timestamp(int(x)))
    control = control[['id', 'userid', 'retweet_id', 'tweet_timestamp', 'retweet_timestamp', 'retweet_userid']]
    control.columns = ['tweetid', 'userid', 'retweet_tweetid', 'tweet_timestamp', 'retweet_timestamp', 'retweet_userid']

    treated['retweet_tweetid'] = treated['retweet_tweetid'].astype(int)
    treated['tweet_timestamp'] = treated['tweetid'].apply(lambda x: get_tweet_timestamp(int(x)))
    treated['retweet_timestamp'] = treated['retweet_tweetid'].apply(lambda x: get_tweet_timestamp(int(x)))

    treated['delta'] = (treated['tweet_timestamp'] - treated['retweet_timestamp']).dt.seconds
    control['delta'] = (control['tweet_timestamp'] - control['retweet_timestamp']).dt.seconds

    cumulative = pd.concat(
        [treated[['userid', 'retweet_userid', 'delta']], control[['userid', 'retweet_userid', 'delta']]])
    cumulative['userid'].astype(int).astype(str)
    cumulative = cumulative.loc[cumulative['delta'] <= timeInterval]

    cumulative = cumulative.groupby(['userid', 'retweet_userid'], as_index=False).count()
    cumulative = cumulative.loc[cumulative['delta'] > 1]

    cum = nx.from_pandas_edgelist(cumulative, 'userid', 'retweet_userid', 'delta')

    cum['userid'].astype(int).astype(str)
    cum = cum.loc[cum['delta'] > 1]

    urls = dict(zip(list(cum.retweet_userid.unique()), list(range(cum.retweet_userid.unique().shape[0]))))
    cum['retweet_userid'] = cum['retweet_userid'].apply(lambda x: urls[x]).astype(int)
    del urls

    userid = dict(zip(list(cum.userid.astype(str).unique()), list(range(cum.userid.unique().shape[0]))))
    cum['userid'] = cum['userid'].astype(str).apply(lambda x: userid[x]).astype(int)

    person_c = CategoricalDtype(sorted(cum.userid.unique()), ordered=True)
    thing_c = CategoricalDtype(sorted(cum.retweet_userid.unique()), ordered=True)

    row = cum.userid.astype(person_c).cat.codes
    col = cum.retweet_userid.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((cum["delta"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
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

    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))

    return G

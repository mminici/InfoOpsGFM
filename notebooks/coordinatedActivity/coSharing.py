import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype


# Data assumptions:
#   - data: Pandas dataframe with columns ['post_id', 'userid', 'feature_shared']   

def coSharing(data):
    temp = data.groupby('feature_shared', as_index=False).count()
    data = data.loc[data['feature_shared'].isin(temp.loc[temp['userid'] > 1]['feature_shared'].to_list())]

    data['value'] = 1

    ids = dict(zip(list(data.feature_shared.unique()), list(range(data.feature_shared.unique().shape[0]))))
    data['feature_shared'] = data['feature_shared'].apply(lambda x: ids[x]).astype(int)
    del ids

    userid = dict(zip(list(data.userid.astype(str).unique()), list(range(data.userid.unique().shape[0]))))
    data['userid'] = data['userid'].astype(str).apply(lambda x: userid[x]).astype(int)

    person_c = CategoricalDtype(sorted(data.userid.unique()), ordered=True)
    thing_c = CategoricalDtype(sorted(data.feature_shared.unique()), ordered=True)

    row = data.userid.astype(person_c).cat.codes
    col = data.feature_shared.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((data["value"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
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

    return G

import pandas as pd
import numpy as np
import networkx as nx

def mergeNetworks(singleFeatureNets, weighted=False):
    """
    Merges multiple networks in a single one, where two nodes will be connected if they are in at least one of the input networks.

    Args:
        singleFeatureNets: List of networks to merge, each network must be a networkx.Graph object
        weighted: boolean variable indicating if the merged network should me weighted or not. If True, multiple weights for the same edge are grouped taking the maximum.
    Returns:
        M: merged network
    """

    graphs = []

    for net in singleFeatureNets:
        if weighted:
            df = pd.DataFrame(net.edges(data='weight'))
        else:
            df = pd.DataFrame(net.edges())
        graphs.append(df)

    temp = pd.concat([df for df in graphs])
    temp = temp.loc[temp[0]!=temp[1]]
    
    if weighted:
        temp.columns = ['source', 'target', 'weight']
        temp = temp.groupby(['source', 'target'], as_index=False).max()
    else:
        temp.columns = ['source', 'target']
        
    temp.dropna(inplace=True)

    if weighted:
        M = nx.from_pandas_edgelist(temp, edge_attr=True)
    else:
        M = nx.from_pandas_edgelist(temp)
        
    return M

def computeCentrality(graphPath):
    """
    Computes the eigenvector centrality for every node in a given network.

    Args:
        graphPath: path of the graph file in .gexf extension
    Returns:
        df: a DataFrame collecting the eigenvector centrality for every user (columns= [userid, eigenvectorCentr])
    """
    G = nx.read_gexf(graphPath)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    
    eigenvector_dict = nx.eigenvector_centrality(G, max_iter=500)
    nx.set_node_attributes(graph, eigenvector_dict, 'eigenvectorCentr')
    
    df = pd.DataFrame(graph.nodes(data='eigenvectorCentr'))
    df.columns = ['userid', 'eigenvectorCentr']
    df['userid'] = df['userid'].astype(str)

    return df
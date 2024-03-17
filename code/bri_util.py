import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from itertools import permutations


def count_rep(person, mapping):
    """Return a dictionary containing the count of meeting attendance
    by organization per each issue
    

    Parameters
    ----------
    person : str
        ID of a stakeholder
    mapping : dict
        Mapping of stakeholder ID to organization ID

    Returns
    -------
    org_counts : dict
        A dictionary containing the org_id and its attendance count

    """
    org_counts = defaultdict(int)
    person_str = str(person)
    # Process if the string is not nan
    if person_str != 'nan':
        pids = person_str.split(',')
        # Relabel person(s) to their respective organization
        pids_relabeled = [mapping[int(pid)] for pid in pids]
        # Start counting the representatives from each organization
        for pid in pids_relabeled:
            org_counts[pid] += 1
    return org_counts


def get_org_name(org_id, mapping):
    """Return the organization name from a given organization ID
    

    Parameters
    ----------
    org_id : int
        Organization ID
    mapping : TYPE
        Mapping of Organization ID to Organization name

    Returns
    -------
    str
        Organization name

    """
    return mapping[org_id]
        

def count_mentions_in_session(df):
    """Return a dictionary containing Organization and their number 
    of mentions in every meeting session
    

    Parameters
    ----------
    minutes : DataFrame
        The dataframe of all issues from the Joint Committee meeting

    Returns
    -------
    session_mentions : list
        A list of dictionaries with organization ID and 
        number of mentions pairs

    """
    session_mentions = []
    df = df[['session', 'mentions']]
    # For each bilateral session, create a dictionary of org_ID and mentions
    for i in range(1,28):
        total_count = defaultdict(int)
        # Extract the bilateral session
        subset = df[df.session == i]
        # mention is a dict 
        for mention in subset.mentions:
            for k, v in mention.items():
                total_count[k] += v
        session_mentions.append(total_count)
    return session_mentions


def create_pairs(session_sth):
    """Return a tuple (org_id, org_id, 1) to represent
    the fact that they are both present in a session
    

    Parameters
    ----------
    session_sth : dict
        Dictionary containing org_id - count pairs

    Returns
    -------
    list
        List of tuples (org_id, org_id, 1)

    """
    oids_in_session = session_sth.keys()
    # We use permutation because the adjacency matrix is symmetric
    # when the graph is undirected
    permu_list = list(permutations(oids_in_session, 2))
    return permu_list


def create_adjmat(ids, actors_count):
    """Create an adjacency matrix from a list of dictionaries. A dictionary
    contains ord_id and attendance pairs
    

    Parameters
    ----------
    ids : Series
        Series of org_id of actors
    actors_count : list
        A list of dictionaries

    Returns
    -------
    adjmat : DataFrame
        Adjacency matrix

    """
    # Initialize an adjacency matrix and prefill with zeros.
    # The index of adjmat is org_id
    n = len(ids)
    adjmat = pd.DataFrame(
    np.zeros((n,n)),
    index = ids,
    columns = ids
    )

    # Fill in adjmat using actors_count
    m = len(actors_count)
    # Iterate through sessions to add relationship score
    for i in range(0, m):
        oid_pairs = create_pairs(actors_count[i])
        # Add a score of 1 to each org pair in the adjacency matrix
        for pair in oid_pairs:
            # The index of adjmat is already by org_id
            adjmat.loc[pair[0], pair[1]] += 1
    
    # Delete organizations that are non-attendees/non-stakeholders 
    # to simplify the adjacency matrix
    col2drop = adjmat.columns[adjmat.sum(axis=1) == 0]
    adjmat.drop(col2drop, inplace = True, axis=0)
    adjmat.drop(col2drop, inplace = True, axis=1)
    return adjmat


def plot_network(G, node_map, attendees, org_id, plot_title):
    """Plots the network graph using the Spring layout and seed = 410
    

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        a graph object created from an adjacency matrix
    node_map : dict
        a mapping from node_id to org_id
    attendees : list or set
        A list of dictionaries. A dictionary contains
        org_id-attendance pair for each session. Alternatively, this 
        function also takes in attendees as a set of org_id
    org_idf : DataFrame
        The org_id dataframe

    Returns
    -------
    None.

    """
    # Create a set of attendees to separating attendees from stakeholders later
    
    if type(attendees) == list:
        att_set = set()
        for att in attendees:
            att_set.update(att.keys())
    elif type(attendees) == set:
        att_set = attendees.copy()
        
    # Categorize the nodes according to their country
    th_attendees = []
    th_sths = []
    cn_attendees = []
    cn_sths = []
    lp_attendees = [] # there's no specific lao stakeholder mentioned

    # Iterate through the nodes and group each node according to their country
    for node_id, node in enumerate(G.nodes()):
        # node_id corresponds to the node index in the graph object.
        # Therefore, we can extract the org_id from each node_id
        oid = node_map[node_id]
        # Match each node by its respective country
        if org_id.loc[org_id['org_id'] == oid, 'country'].values[0] == 'th':
            if oid in att_set:
                th_attendees.append(node)
            else:
                th_sths.append(node)
        elif org_id.loc[org_id['org_id'] == oid, 'country'].values[0] == 'cn':
            if oid in att_set:
                cn_attendees.append(node)
            else:
                cn_sths.append(node)
        else:
            lp_attendees.append(node)
    
    # Plot the network graph
    pos = nx.spring_layout(G, weight='none', seed=410) #seed=42
    fig, ax = plt.subplots(figsize = (11,7), dpi=300)
    nx.draw_networkx(
        G,
        pos = pos,
        node_size = 0,
        width = 0.5,
        with_labels = True,
        labels = node_map,
        font_size=9,
        font_color = 'white',
        font_weight = 'semibold',
        edge_color = 'gainsboro',
        ax = ax
        )
    # Solid square nodes for the thai attendees
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist = th_attendees,
        node_size=500,
        node_shape = 's'
        )
    # Light square nodes for the thai stakeholders
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist = th_sths,
        node_shape = 's',
        node_size=500,
        alpha = 0.5
        )
    # Solid circle nodes for the chinese attendees
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist = cn_attendees,
        node_shape = 'o',
        node_size=500
        )
    # Light circle nodes for the chinese stakeholders
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist = cn_sths,
        node_shape = 'o',
        node_size=500,
        alpha = 0.5
        )
    # Solid diamond nodes for the lao attendees
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist = lp_attendees,
        node_shape = 'd',
        node_size = 650
        )
    plt.title(plot_title)
    plt.show()
    
    

def plot_network2(G, node_map, attendees, org_id, plot_title, node2abbre):
    """Plots the network graph using the Spring layout and seed = 410
 

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        a graph object created from an adjacency matrix
    node_map : dict
        a mapping from node_id to org_id
    attendees : list or set
        A list of dictionaries. A dictionary contains
        org_id-attendance pair for each session. Alternatively, this 
        function also takes in attendees as a set of org_id
    org_idf : DataFrame
        The org_id dataframe

    Returns
    -------
    None.

    """
    # Create a set of attendees to separating attendees from stakeholders later
    
    if type(attendees) == list:
        att_set = set()
        for att in attendees:
            att_set.update(att.keys())
    elif type(attendees) == set:
        att_set = attendees.copy()
        
    # Categorize the nodes according to their country
    th_attendees = []
    th_sths = []
    cn_attendees = []
    cn_sths = []
    lp_attendees = [] # there's no specific lao stakeholder mentioned

    # Iterate through the nodes and group each node according to their country
    for node_id, node in enumerate(G.nodes()):
        # node_id corresponds to the node index in the graph object.
        # Therefore, we can extract the org_id from each node_id
        oid = node_map[node_id]
        # Match each node by its respective country
        if org_id.loc[org_id['org_id'] == oid, 'country'].values[0] == 'th':
            if oid in att_set:
                th_attendees.append(node)
            else:
                th_sths.append(node)
        elif org_id.loc[org_id['org_id'] == oid, 'country'].values[0] == 'cn':
            if oid in att_set:
                cn_attendees.append(node)
            else:
                cn_sths.append(node)
        else:
            lp_attendees.append(node)
    
    # Plot the network graph
    pos = nx.spring_layout(G, weight='none', seed=410) #seed=410
    fig, ax = plt.subplots(figsize = (14,14), dpi=400)
    nx.draw_networkx(
        G,
        pos = pos,
        node_size = 25,
        width = 0.5,
        #edge_color = '808080', ##808080
        with_labels = True,
        labels = node2abbre,
        font_size = 13,
        #font_color = 'k',
        font_weight = 'bold',
        ax = ax
        )
    # # Solid square nodes for the thai attendees
    # nx.draw_networkx_nodes(
    #     G,
    #     pos,
    #     nodelist = th_attendees,
    #     node_shape = 's'
    #     )
    # # Light square nodes for the thai stakeholders
    # nx.draw_networkx_nodes(
    #     G,
    #     pos,
    #     nodelist = th_sths,
    #     node_shape = 's',
    #     alpha = 0.35
    #     )
    # # Solid circle nodes for the chinese attendees
    # nx.draw_networkx_nodes(
    #     G,
    #     pos,
    #     nodelist = cn_attendees,
    #     node_shape = 'o'
    #     )
    # # Light circle nodes for the chinese stakeholders
    # nx.draw_networkx_nodes(
    #     G,
    #     pos,
    #     nodelist = cn_sths,
    #     node_shape = 'o',
    #     alpha = 0.35
    #     )
    # # Solid diamond nodes for the lao attendees
    # nx.draw_networkx_nodes(
    #     G,
    #     pos,
    #     nodelist = lp_attendees,
    #     node_shape = 'd'
    #     )
    # plt.title(plot_title)
    # # plt.style.use('grayscale')
    # plt.show()

    

def plot_network_score(df, score_type):
    """Create two plots of the closeness score. A plot is for Thai attendees
    and the other plot is for Chinese attendees

    Parameters
    ----------
    df : DataFrame
        Dataframe containing org_name, and score. The index is org_id
    score_type : str
        Text describing the score type: 'Closeness' or 'Betweenness'

    Returns
    -------
    None.

    """
    # Subset of Thai attendees
    df_copy = df.copy().reset_index().rename({'index': 'org_id'}, axis=1)
    # The range function exclude the end value
    mask_th = [k in range(0, 200) for k in df_copy.org_id]
    thai_df = df_copy.loc[mask_th, :]
    # Subset of Chinese Attendees
    df_copy = df.copy().reset_index().rename({'index': 'org_id'}, axis=1)
    mask_cn = [k in range(200, 300) for k in df_copy.org_id]
    china_df = df_copy.loc[mask_cn, :]
    # Plot the both sides separately, starting with Thailand
    fig, ax = plt.subplots(1, 2, figsize=(13,5))
    # plt.style.use('grayscale')
    ax[0].barh(
        thai_df['org_name'],
        thai_df['score']
        )
    ax[0].set_xlabel(score_type + ' scores')
    # Make the top side and right side of box invisible
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].bar_label(ax[0].containers[0], 
                    fmt = '%.3f',
                    label_type = 'edge')
    # China side
    ax[1].barh(
        china_df['org_name'],
        china_df['score']
        )
    ax[1].set_xlabel(score_type + ' scores')
    # Make the top side and right side of box invisible
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].bar_label(ax[1].containers[0],
                    fmt = '%.3f',
                    label_type = 'edge')
    plt.tight_layout()
    plt.show()


def plot_network_score_all(df, score_type):
    """Create two plots of the closeness score. A plot is for Thai attendees
    and the other plot is for Chinese attendees

    Parameters
    ----------
    df : DataFrame
        Dataframe containing org_name, and score. The index is org_id
    score_type : str
        Text describing the score type: 'Closeness' or 'Betweenness'

    Returns
    -------
    None.

    """
    # Subset of Thai attendees
    df_copy = df.copy().reset_index().rename({'index': 'org_id'}, axis=1)
    # The range function exclude the end value
 
    # Plot the both sides separately, starting with Thailand
    fig, ax = plt.subplots(figsize=(13,5))
    # plt.style.use('grayscale')
    ax.barh(
        df_copy['org_name'],
        df_copy['score']
        )
    ax.set_xlabel(score_type + ' scores')
    # Make the top side and right side of box invisible
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.bar_label(ax.containers[0], 
                    fmt = '%.3f',
                    label_type = 'edge')

    plt.tight_layout()
    plt.show()


def plot_network_score_TH(df, score_type):
    """Plot one network score barchart

    Parameters
    ----------
    df : DataFrame
        Dataframe containing org_name, and score. The index is org_id
    score_type : str
        Text describing the score type: 'Closeness' or 'Betweenness'

    Returns
    -------
    None.

    """
    # Subset of Thai attendees
    df_copy = df.copy().reset_index().rename({'index': 'org_id'}, axis=1)
    # The range function exclude the end value
    mask_th = [k in range(0, 200) for k in df_copy.org_id]
    thai_df = df_copy.loc[mask_th, :]
    # Subset of Chinese Attendees
    df_copy = df.copy().reset_index().rename({'index': 'org_id'}, axis=1)
    mask_cn = [k in range(200, 300) for k in df_copy.org_id]
    china_df = df_copy.loc[mask_cn, :]
    # Plot the both sides separately, starting with Thailand
    fig, ax = plt.subplots(figsize=(7,5))

    ax.barh(
        thai_df['org_name'],
        thai_df['score']
        )
    ax.set_xlabel(score_type + ' scores')
    # Make the top side and right side of box invisible
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.bar_label(ax.containers[0], 
                    fmt = '%.3f',
                    label_type = 'edge')
    plt.tight_layout()
    plt.show()


def count_rep_rs(person):
    """Return a dictionary containing the count of meeting attendance
    by organization per each issue
    

    Parameters
    ----------
    person : str
        ID of a stakeholder
    mapping : dict
        Mapping of stakeholder ID to organization ID

    Returns
    -------
    org_counts : dict
        A dictionary containing the org_id and its attendance count

    """
    org_counts = defaultdict(int)
    person_str = str(person)
    # Process if the string is not nan
    if person_str != 'nan':
        pids = person_str.split(',')
        # Start counting the representatives from each organization
        for pid in pids:
            org_counts[pid] += 1
    return org_counts


def concat_sender_receiver(rs_data, senderRecipient):
    """Concatenate sender and receiver columns to rs_data
    

    Parameters
    ----------
    rs_data : DataFrame
        Dataframe from resolution_data.xlsx
    senderRecipient : DataFrame
        Dataframe from resolution_sender_recipient.xlsx

    Returns
    -------
    rs_data : DataFrame
        rs_data with two new columns

    """
    # Use a copy version to avoid editing the original dataframe
    rs_data = rs_data.copy()
    # Appender sender, receiver, sender_abbre columns
    letter2sender = pd.Series(
        senderRecipient['sender'].values,
        index = senderRecipient['letter_id'].values
        ).to_dict()
    rs_data['sender'] = rs_data['letter_id'].apply(
        lambda letter_id, mapping : mapping[str(letter_id)],
        args = ((letter2sender), )
        )
    # Map letter_id to Receiver
    letter2sender = pd.Series(
        senderRecipient['receiver'].values,
        index = senderRecipient['letter_id'].values
        ).to_dict()
    rs_data['receiver'] = rs_data['letter_id'].apply(
        lambda letter_id, mapping : mapping[str(letter_id)],
        args = ((letter2sender), )
        )
    return rs_data
    

def rename_scores(score_dict, node_map):
    """Rename the keys of a dictionary from node_id to org_id
    

    Parameters
    ----------
    score_dict : dict
        the output from nx.betweenness_centrality or nx.closeness_centrality
    node_map : dict
        a map from node_id to org_id

    Returns
    -------
    renamed_dict : dict
        renamed dictionary of the centrality score

    """
    renamed_dict = {}
    # Map the node to org_id
    for k, v in score_dict.items():
        oid = node_map[k]
        renamed_dict[oid] = v
    return renamed_dict
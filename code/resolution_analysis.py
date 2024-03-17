from bri_util import *
import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


#%% Read the resolution data files
data_path = os.path.abspath(r'..\\Data\\')
rs_data_raw = pd.read_excel(
    os.path.join(
        data_path, 'resolution_data.xlsx'
        )
    )

senderRecipient = pd.read_excel(
    os.path.join(
        data_path, 'resolution_sender_recipient.xlsx'
        ),
    usecols = list(range(0,4))
    )

org_id = pd.read_excel(
    os.path.join(data_path, 'org_id.xlsx'),
    usecols = list(range(0,6))
    )


#%% Plot number of letters vs resolution ID

senderRecipient['resolution_id'].value_counts(sort=False) \
    .plot(
        xlabel = 'Resolution ID',
        ylabel = 'Letter count',
        kind = 'bar'
        )
plt.show()



#%% Plot the issue count by issue type
rs_data = rs_data_raw[rs_data_raw['issue_id'] != 0].copy()
rs_data.issue_id.replace(
    {
     1:'Project design',
     2:'Technical',
     3:'Financial',
     4:'Legal'
     },
    inplace=True
    )

rs_data.issue_id.value_counts().plot(
    kind  ='bar',
    ylabel = 'Counts',
    title = 'Number of key passage by category')
plt.xticks(rotation=45)
plt.show()


#%% Plot issues by resolution_id
# Concatenate sender & receiver columns to rs_data
rs_data = concat_sender_receiver(rs_data, senderRecipient)

# Map org_id to its abbreviation
abbre_mapping = pd.Series(
    org_id.abbre.values,
    index = org_id.org_id.values
    ).to_dict()


# Create a new dataframe by merging resolution data and senderRecipient
rs_data['sender_abbre'] = rs_data['sender'].apply(
        lambda x, y : y[int(x)],
        args = ((abbre_mapping), )
        )
    
# Create a new df with the last column being the number of each issue type
session_df = rs_data[['session', 'issue_id']] \
    .groupby(['session', 'issue_id']) \
        .size() \
            .reset_index()
# Pivot session_df to have row be a session and column be an issue ID
session_df = session_df.pivot(
    index='session', 
    columns='issue_id', 
    values=0
    )
# Sum across columns to get the total number of issues per session
session_df['total'] = session_df.sum(axis=1)
session_df.reset_index(inplace=True)
session_df.fillna(0, inplace=True)

# Create a stacked bar plot for the number of issues vs meeting session
fig, ax = plt.subplots(figsize=(7,5))

ax.bar(
       session_df['session'],
       session_df['Project design'], 
       label = 'Project design',
       color = '#cccccc'
       )
ax.bar(
       session_df['session'],
       session_df['Technical'],
       bottom = session_df['Project design'], 
       label = 'Technical',
       color = '#969696'
       )
ax.bar(
       session_df['session'],
       session_df['Financial'],
       bottom = session_df['Project design'] \
           + session_df['Technical'], 
       label = 'Financial',
       color = '#636363'
       )
ax.bar(
       session_df['session'],
       session_df['Legal'],
       bottom = session_df['Project design'] \
           + session_df['Technical'] + session_df['Financial'], 
       label = 'Legal',
       color = '#252525'
       )
# Format the stacked bar plot
ax.set_ylabel('Key Passage Count')
ax.set_xlabel('Resolution ID')
ax.set_xlim(0, 17)
plt.xticks(session_df['session'])
ax.set_title('Key Passages vs Resolution ID')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='best')
plt.show()


#%% Plot issues by organization using stacked bar graph
issuesBySenders = rs_data.pivot_table(
    index = 'sender_abbre',
    columns = 'issue_id', 
    aggfunc = 'size',
    fill_value = 0
    )
# Reorder the columns
issuesBySenders = issuesBySenders[
    ['Project design', 'Technical', 'Financial', 'Legal']
    ].reset_index()
# Create a column of total issue count and sort
issuesBySenders['total'] = issuesBySenders.sum(axis=1)
issuesBySenders.sort_values('total', ascending=True, inplace=True)
# Stacked barh plot
fig, ax = plt.subplots(figsize=(7,5))

ax.barh(
       issuesBySenders['sender_abbre'],
       issuesBySenders['Project design'], 
       label = 'Project design',
       color = '#cccccc'
       )
ax.barh(
       issuesBySenders['sender_abbre'],
       issuesBySenders['Technical'],
       left = issuesBySenders['Project design'], 
       label = 'Technical',
       color = '#969696'
       )
ax.barh(
       issuesBySenders['sender_abbre'],
       issuesBySenders['Financial'],
       left = issuesBySenders['Project design'] \
           + issuesBySenders['Technical'], 
       label = 'Financial',
       color = '#636363'
       )
ax.barh(
       issuesBySenders['sender_abbre'],
       issuesBySenders['Legal'],
       left = issuesBySenders['Project design'] \
           + issuesBySenders['Technical'] + issuesBySenders['Financial'], 
       label = 'Legal',
       color = '#252525'
       )
# Format the stacked bar plot
ax.set_xlabel('Key Passage count')
ax.set_title('Key Passages by agency')
ax.legend(loc='best')
plt.show()


#%% Plot the number of organization mentions
# Create a new column containing a dictionary of mentions
rs_data['mentions'] = rs_data['stakeholders_mentioned']\
    .apply(count_rep_rs)
    
# Create a dataframe cols = org_id and rows = session_id.
# The entries represent the number of mentions
session_mentions = pd.DataFrame(
    count_mentions_in_session(rs_data)
    ).fillna(0).rename(abbre_mapping)

# Drop columns with only zeros
session_mentions = session_mentions.loc[
    :, (session_mentions != 0).any(axis=0)
    ]
# Create a total column
session_mentions['total'] = session_mentions.sum(axis=1)

# Plot the result
session_mentions['total'] \
    .plot(
        kind='bar',
        xlabel='Session',
        ylabel='Total mentions',
        title='Number of stakeholder mention in a session'
        )
plt.show()


#%% Plot stakeholder vs mention type
# Extract a set of unique stakeholders
stakeholders = set(session_mentions.columns)
# Don't need to include total
stakeholders.remove('total')
# Initialize a counting dataframe
issue_mention = pd.DataFrame(
    0,
    index = [
        'Project design', 'Technical', 'Financial', 'Legal'
        ],
    columns = stakeholders)

# Iterate through the rows of issues
for idx, row in rs_data.iterrows():
    for oid in row['mentions'].keys():
        # Some issue does not have an issue_id
        if not pd.isna(rs_data.loc[idx, 'issue_id']):
            issue_mention.loc[rs_data.loc[idx, 'issue_id'], oid] += 1

# Rename the columns of issue_mention to org_name
org_mapping = pd.Series(
    org_id.name.values,
    index = org_id.org_id.values.astype(str)
    ).to_dict()
issue_mention.rename(org_mapping, axis=1, inplace=True)

# Transpose to bring org_name to rows
issue_mention = issue_mention.T

fig, ax = plt.subplots(figsize=(7,5))

ax.barh(
       issue_mention.index,
       issue_mention['Project design'], 
       label = 'Project design',
       color = '#cccccc'
       )
ax.barh(
       issue_mention.index,
       issue_mention['Technical'],
       left = issue_mention['Project design'], 
       label = 'Technical',
       color = '#969696'
       )
ax.barh(
       issue_mention.index,
       issue_mention['Financial'],
       left = issue_mention['Project design'] \
           + issue_mention['Technical'], 
       label = 'Financial',
       color = '#636363'
       )
ax.barh(
       issue_mention.index,
       issue_mention['Legal'],
       left = issue_mention['Project design'] \
           + issue_mention['Technical'] + issue_mention['Financial'], 
       label = 'Legal',
       color = '#252525'
       )
# Format the stacked bar plot
ax.set_xlabel('Key Passage count')
ax.set_title('Stakeholders vs issue type')
ax.legend(loc='best')
plt.show()


#%% Plot the network graph for senders & receivers only
rs_adjmat = pd.DataFrame(
    0, index = org_id.org_id, columns = org_id.org_id
    )
# Iterate through the rows senderRecipient and add score to the pairs
for idx, _  in senderRecipient.iterrows():
    # Update the sender recipient, the undirected graph is symmetric
    sender_id = senderRecipient.loc[idx, 'sender']
    receive_id = senderRecipient.loc[idx, 'receiver']
    rs_adjmat.loc[sender_id, receive_id] += 1
    rs_adjmat.loc[receive_id, sender_id] += 1

# Drop non-attendees to simplyfy the adjacency matrix
col2drop = rs_adjmat.columns[rs_adjmat.sum(axis=1) == 0]
rs_adjmat.drop(col2drop, inplace = True, axis=0)
rs_adjmat.drop(col2drop, inplace = True, axis=1)

# Map the nodes to their respective org_id
node_map = {k: v for k, v in enumerate(rs_adjmat.columns)}

# Map org_id to its abbreviation
abbre_mapping = pd.Series(
    org_id.abbre.values,
    index = org_id.org_id.values
    ).to_dict()

# Map nodes to its abbreviation
node2abbre = {}
for k in node_map.keys():
    node2abbre[k] = abbre_mapping[node_map[k]]

# Create a graph object from adjmat
G = nx.from_numpy_matrix(rs_adjmat.values)

# Identify the attendees so their nodes will appear differently from stakeholders
attendees = set(senderRecipient.sender)
attendees.update(senderRecipient.receiver)

# Plot the network graph using the graph object G
# plot_network(
#     G, node_map, attendees, org_id,
#     'Internal Consultation Stage - Only Attendees'
#     )


#%% Plot the network graph for all actors
rs_adjmat_actors = pd.DataFrame(
    0, index = org_id.org_id, columns = org_id.org_id
    )
# rs_data_raw has not dropped issues with issue_type = 0
rs_data_raw = concat_sender_receiver(rs_data_raw, senderRecipient)

# Map the nodes to their respective org_id
node_map = {k: v for k, v in enumerate(rs_adjmat_actors.columns)}

# Map org_id to its abbreviation
abbre_mapping = pd.Series(
    org_id.abbre.values,
    index = org_id.org_id.values
    ).to_dict()


# Iterate through each letter_id and update the adjacency matrix
for idx, _  in senderRecipient.iterrows():
    # First, Update the sender recipient. Note that he undirected graph 
    # is symmetric.
    sender_id = senderRecipient.loc[idx, 'sender']
    receive_id = senderRecipient.loc[idx, 'receiver']
    rs_adjmat_actors.loc[sender_id, receive_id] += 1
    rs_adjmat_actors.loc[receive_id, sender_id] += 1
    # Second, update the mentions by iterate through rows of rs_data
    letter_id = senderRecipient.loc[idx, 'letter_id']
    sub_df = rs_data_raw.loc[
            rs_data_raw['letter_id'] == letter_id, ['sender', 'stakeholders_mentioned']
            ]
    for _, row in sub_df.iterrows():
        sender = row.loc['sender']
        stakeholders = str(row.loc['stakeholders_mentioned'])
        if stakeholders != 'nan':
            for stakeholder in stakeholders.split(','):
                rs_adjmat_actors.loc[sender, int(stakeholder)] += 1
                rs_adjmat_actors.loc[int(stakeholder), sender] += 1

# Drop non-attendees to simplyfy the adjacency matrix
col2drop = rs_adjmat_actors.columns[rs_adjmat_actors.sum(axis=1) == 0]
rs_adjmat_actors.drop(col2drop, inplace = True, axis=0)
rs_adjmat_actors.drop(col2drop, inplace = True, axis=1)

# Map the nodes to their respective org_id
node_map_actors = {k: v for k, v in enumerate(rs_adjmat_actors.columns)}

# Create a graph object from adjmat
G_actors = nx.from_numpy_matrix(rs_adjmat_actors.values)

# Map nodes to its abbreviation
node2abbre = {}
for k in node_map_actors.keys():
    node2abbre[k] = abbre_mapping[node_map_actors[k]]

# Plot the network graph
plot_network(
    G_actors,
    node_map_actors,
    attendees,
    org_id,
    'Internal Consultation Stage - All Actors')


#%% Calculate the betweenness and closeness scores
# Calculate the betweenness and closeness scores
btwn_score = nx.betweenness_centrality(G)
clsn_score = nx.closeness_centrality(G)

# Rename from node_id to org_id
btwn_score_dict = rename_scores(btwn_score, node_map)
clsn_score_dict = rename_scores(clsn_score, node_map)

# Transform dict to dataframe and sort based on the scores
btwn_score_df = pd.DataFrame(btwn_score_dict, index=['score']).T
btwn_score_df.sort_values('score', ascending=True, inplace=True)
btwn_score_df['org_name'] = [abbre_mapping[k] for k in btwn_score_df.index]

clsn_score_df = pd.DataFrame(clsn_score_dict, index=['score']).T
clsn_score_df.sort_values('score', ascending=True, inplace=True)
clsn_score_df['org_name'] = [abbre_mapping[k] for k in clsn_score_df.index]


#%% #%% Plot betweeness and closeness scores
# Betweenness
plot_network_score_TH(btwn_score_df, score_type='Betweenness')
# Closeness
plot_network_score_TH(clsn_score_df, score_type='Closeness')

# Calculate the density
print('Density of internal network is ', round(nx.density(G), 4))
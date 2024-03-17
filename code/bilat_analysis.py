import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from bri_util import *
from collections import defaultdict


#%% Read the data files
data_path = os.path.abspath(r'..\\Data\\')

stakeholder_id = pd.read_excel(
    os.path.join(data_path, 'stakeholder_id.xlsx'),
    usecols = list(range(0,5))
    )

org_id = pd.read_excel(
    os.path.join(data_path, 'org_id.xlsx'),
    usecols = list(range(0,6))
    )

meeting_info = pd.read_excel(
    os.path.join(data_path, 'meeting_info.xlsx'),
    usecols = list(range(0,7)),
    dtype=str
    ) 

minutes = pd.read_excel(
    os.path.join(data_path, 'minutes.xlsx'),
    usecols = list(range(1,9))
    )

issue_id = pd.read_excel(
    os.path.join(data_path, 'issue_id.xlsx'),
    usecols = list(range(0,3))
    )

project_id = pd.read_excel(
    os.path.join(data_path, 'project_id.xlsx'),
    usecols = list(range(0,3))
    )


#%% Process stakeholder_id
# Map stakeholder_id to org_id
stakeholder_mapping = pd.Series(
    stakeholder_id['org_id'].values,
    index = stakeholder_id['stakeholder_id'].values
    ).to_dict()


#%% Process minutes
# Create a new column with a dict of org_id and count pairs
minutes['mentions'] = minutes['stakeholders_mentioned'] \
    .apply(
        count_rep, args=((stakeholder_mapping),)
        )

# Convert the thai_part and chinese_party to dictionaries of org counts
meeting_info['thai_party'] = meeting_info['thai_party'] \
    .apply(
        count_rep, args=((stakeholder_mapping),)
        )

meeting_info['chinese_party'] = meeting_info['chinese_party'] \
    .apply(
        count_rep, args=((stakeholder_mapping),)
           )

meeting_info['other'] = meeting_info['other'] \
    .apply(
        count_rep,args=((stakeholder_mapping),)
        )

# Create a new column in meeting_info with all attendees
thai_party = meeting_info['thai_party'].values.copy()
chinese_party = meeting_info['chinese_party'].values.copy()
lao_party = meeting_info['other'].values.copy()
attendees = []
for i in range(0, 27):
    attendees.append(
        {
            **thai_party[i],
            **chinese_party[i],
            **lao_party[i]}
        )

meeting_info['attendees'] = pd.Series(attendees)


#%% Count the number of meetings per each location
print('\nFrequency and location of meetings')
print(meeting_info.location.value_counts())


#%% Plot the number of mentions by issue_id
minutes.issue_id.replace(
    {
     1:'Project design',
     2:'Technical',
     3:'Financial',
     4:'Legal'
     },
    inplace=True
    )

minutes.issue_id.value_counts() \
    .plot(
        kind  ='bar',
        xlabel = 'Issue ID',
        ylabel = 'Counts',
        title = 'Number of key passage by category'
        )
plt.xticks(rotation=45)
plt.show()

#%% Plot the count of issues per session
# Create a new df with the last column being the number of each issue type
session_df = minutes[['session', 'issue_id']] \
    .groupby(['session', 'issue_id']) \
        .size() \
            .reset_index()
# Pivot session_df to have a row be a session and a column be an issue ID
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
# plt.style.use('grayscale')
ax.bar(
       session_df['session'],
       session_df.iloc[:, 1], 
       label = 'Project design',
       color = '#cccccc'
       )
ax.bar(
       session_df['session'],
       session_df.iloc[:, 2],
       bottom = session_df.iloc[:, 1], 
       label = 'Technical',
       color = '#969696'
       )
ax.bar(
       session_df['session'],
       session_df.iloc[:, 3],
       bottom = session_df.iloc[:, 1] + session_df.iloc[:, 2], 
       label = 'Financial',
       color = '#636363'
       )
ax.bar(
       session_df['session'],
       session_df.iloc[:, 4],
       bottom = session_df.iloc[:, 1] + session_df.iloc[:, 2] + session_df.iloc[:, 3], 
       label = 'Legal',
       color = '#252525'
       )
# Format the stacked bar plot
ax.set_ylabel('Count')
ax.set_xlabel('Meeting session')
ax.set_xlim(0, 28)
ax.set_ylim(0, 26)
ax.set_title('Number of key passage by category')
ax.legend(loc='best')
plt.show()


#%% Plot the number of attendance by organization
# Map org_id to its namee
org_mapping = pd.Series(
    org_id.name.values,
    index = org_id.org_id.values
    ).to_dict()

# Count the number of attendance by each organization as a dict
total_attd_count = defaultdict(int)
for reps in attendees:
    for k, v in reps.items():
        total_attd_count[k] += 1 # Plus 1 to avoid duplicate

# Convert from dict to dataframe fot plotting
total_attd_count = pd.Series(total_attd_count) \
    .reset_index() \
    .rename(
        {'index': 'org_id', 0:'attendance_count'},
        axis = 1
        )
    
# Sort by attendance count
total_attd_count.sort_values('attendance_count', inplace=True)

# Concatenate org_name to total_attd_count
total_attd_count['name'] = total_attd_count['org_id']\
    .apply(
        get_org_name,
        args = ((org_mapping), )
        )

# Plot a barchart of Organization vs attendance
fig, ax = plt.subplots(figsize=(7,5))
plt.barh(
    total_attd_count['name'],
    total_attd_count['attendance_count']
    )
ax.set_xlabel('Count')
ax.set_title('Attendance by organization')
plt.show()


#%% Plot the number of organization mentions
# Create a dataframe cols = org_id and rows = session_id.
# The entries represent the number of mentions
session_mentions = pd.DataFrame(
    count_mentions_in_session(minutes)
    ).fillna(0)

# Plot the result
session_mentions['total'] = session_mentions.sum(axis=1)
session_mentions['total'] \
    .plot(
        kind='bar',
        xlabel='Session',
        ylabel='Total mentions',
        title='Number of stakeholder mention in a session'
        )
plt.show()

# Print the Top 5 stakeholders who were mentioned the most
topmentions = session_mentions.sum(axis=0).sort_values(ascending=False)
topmentions.rename(org_mapping, axis=0, inplace=True)
print('\nTop mentioned stakeholders in the bilateral negotiation stage')
print(topmentions[1:6])

#%% Create a series containing a dict count of all attendees
#   and stakeholders. This step is for the final network analysis
# Create a dict of session mentions 
session_mentions_dict = count_mentions_in_session(minutes)
# Create a list containing dictionaries of stakeholder-mentions pair
actors_count = []
for i in range(0, 27):
    session_sth_dict = defaultdict(int)
    # First, tally the mentions per organization
    for k, v in session_mentions_dict[i].items():
        session_sth_dict[k] += v
    # Second, tally the number of attendence
    for k, v in thai_party[i].items():
        session_sth_dict[k] += v
    for k, v in chinese_party[i].items():
        session_sth_dict[k] += v
    for k, v in lao_party[i].items():
        session_sth_dict[k] += v
    # Append the tally dictionary to actors_countlist (list)
    actors_count.append(session_sth_dict)


#%% Create the adjacency matrix
oids = org_id.org_id
adjmat = create_adjmat(ids=oids, actors_count=actors_count)

# Delete organizations that are non-attendees/non-stakeholders 
# to simplify the adjacency matrix
col2drop = adjmat.columns[adjmat.sum(axis=1) == 0]
adjmat.drop(col2drop, inplace = True, axis=0)
adjmat.drop(col2drop, inplace = True, axis=1)


#%% Plot the network graph
# Map the nodes to their respective org_id
node_map = {k: v for k, v in enumerate(adjmat.columns)}

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
G = nx.from_numpy_matrix(adjmat.values)

# Plot the network graph
from bri_util import plot_network
plot_network(
    G, 
    node_map, 
    attendees, 
    org_id, 
    'Bilateral Negotiation Stage - All Actors')


#%% Compute Closeness and betweeness scores
# Create an adjacency matrix with only attendees
adjmat_attendees = create_adjmat(
    ids = oids,
    actors_count = attendees
    )

# Map the nodes to their respective org_id
node_map_attendees = {k: v for k, v in enumerate(adjmat_attendees.columns)}

# Create a graph object with only attendees to calculate the network scores
G_attendees = nx.from_numpy_matrix(adjmat_attendees.values)

# Calculate the betweenness and closeness scores
btwn_score = nx.betweenness_centrality(G_attendees)
clsn_score = nx.closeness_centrality(G_attendees)

btwn_score_dict = {}
# Map the node to org_id
for k, v in btwn_score.items():
    oid = node_map_attendees[k]
    btwn_score_dict[oid] = v

clsn_score_dict = {}
for k, v in clsn_score.items():
    oid = node_map_attendees[k]
    clsn_score_dict[oid] = v

# Map org_id to its abbreviation
abbre_mapping = pd.Series(
    org_id.abbre.values,
    index = org_id.org_id.values
    ).to_dict()

# Format to dataframe and sort
btwn_score_df = pd.DataFrame(btwn_score_dict, index=['score']).T
btwn_score_df.sort_values('score', ascending=True, inplace=True)
btwn_score_df['org_name'] = [abbre_mapping[k] for k in btwn_score_df.index]

clsn_score_df = pd.DataFrame(clsn_score_dict, index=['score']).T
clsn_score_df.sort_values('score', ascending=True, inplace=True)
clsn_score_df['org_name'] = [abbre_mapping[k] for k in clsn_score_df.index]


#%% Plot the network graph for attendees only
plot_network(G_attendees, 
             node_map_attendees,
             attendees, 
             org_id,
             'Bilateral Negotiation Stage - Only Attendees')


#%% Plot betweeness and closeness scores
# # Betweenness
# plot_network_score(btwn_score_df, score_type='Betweenness')
# # Closeness
# plot_network_score(clsn_score_df, score_type='Closeness')


#%% Centrality scores when using graph with all actors
btwn_score = nx.betweenness_centrality(G)
clsn_score = nx.closeness_centrality(G)
# Rename from node_id to org_id
btwn_score_dict = rename_scores(btwn_score, node_map)
clsn_score_dict = rename_scores(clsn_score, node_map)
# Format to dataframe and sort
btwn_score_df = pd.DataFrame(btwn_score_dict, index=['score']).T
btwn_score_df.sort_values('score', ascending=True, inplace=True)
btwn_score_df['org_name'] = [abbre_mapping[k] for k in btwn_score_df.index]
btwn_score_df = btwn_score_df.loc[
    btwn_score_df.index.isin(adjmat_attendees.index)
    ]

clsn_score_df = pd.DataFrame(clsn_score_dict, index=['score']).T
clsn_score_df.sort_values('score', ascending=True, inplace=True)
clsn_score_df['org_name'] = [abbre_mapping[k] for k in clsn_score_df.index]
clsn_score_df = clsn_score_df.loc[
    clsn_score_df.index.isin(adjmat_attendees.index)
    ]

# Betweenness
plot_network_score_all(btwn_score_df, score_type='Betweenness')
# Closeness
plot_network_score_all(clsn_score_df, score_type='Closeness')

# Calculate the density
print('Density of bilateral network is ', round(nx.density(G), 4))

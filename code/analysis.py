# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 23:36:59 2021 by Phumthep
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


#------ Count the number of meetings
print('\nFrequency and location of meetings')
print(meeting_info.location.value_counts())

#------ Count the number of mentions by issue_id
minutes.issue_id.value_counts().plot(kind  ='bar',
                                     xlabel = 'Issue ID',
                                     ylabel = 'Counts',
                                     title = 'Number of issues categorized by issue_id')
plt.show()

#------ Plot the number of issues by meeting session
df01 = minutes[['session', 'issue_id']]\
    .groupby(['session', 'issue_id'])\
        .size()\
            .reset_index()
df01 = df01.pivot(index='session', columns='issue_id', values=0)
df01['total'] = df01.sum(axis=1)
df01.reset_index(inplace=True)
df01.fillna(0, inplace=True)

fig, ax = plt.subplots(figsize=(7,5))
ax.bar(df01['session'], df01[1], 
       label='project_design')
ax.bar(df01['session'], df01[2], bottom=df01[1], 
       label='technical')
ax.bar(df01['session'], df01[3], bottom=df01[1]+df01[2], 
       label='financial')
ax.bar(df01['session'], df01[4], bottom=df01[1]+df01[2]+df01[3], 
       label='legal')

ax.set_ylabel('Count')
ax.set_xlabel('Meeting session')
ax.set_xlim(0, 28)
ax.set_ylim(0, 26)
ax.set_title('Number of issues in each meeting session categorized by issue id')
ax.legend(loc='best')
plt.show()

#----- Plot the number of attendance by organization
# Creating a mapping between org_id and its name
total_attd_count = defaultdict(int)
for reps in attendees:
    for k, v in reps.items():
        total_attd_count[k] += v


# Format the dataframe
total_attd_count = pd.Series(total_attd_count)\
    .reset_index()\
    .rename({'index': 'org_id',
             0:'attendance_count'},
            axis = 1)
    
org_mapping = pd.Series(org_id.name.values,
                        index = org_id.org_id.values)\
    .to_dict()

total_attd_count['name'] = total_attd_count['org_id']\
    .apply(get_org_name,
           args=((org_mapping), ))

fig, ax = plt.subplots(figsize=(7,5))
plt.barh(total_attd_count['name'],
        total_attd_count['attendance_count'])
ax.set_xlabel('Count')
ax.set_title('Total number of attendance by organization')
plt.show()

#----- Plot the number of organization mentions
session_mentions = pd.DataFrame(
    count_mentions_in_session(minutes)
    ).fillna(0)

session_mentions['total'] = session_mentions.sum(axis=1)
session_mentions['total'].plot(kind='bar',
                               xlabel='Session',
                               ylabel='Total mentions',
                               title='Number of stakeholder mentions in a session')
plt.show()

# Top 5 stakeholders who were mentioned the most
topmentions = session_mentions.sum(axis=0).sort_values(ascending=False)
topmentions.rename(org_mapping, axis=0, inplace=True)
print('\nTop mentioned stakeholders')
print(topmentions[1:6])

#----- Combine all stakeholders
# Create a dict of session mentions
session_mentions_dict = count_mentions_in_session(minutes)
# Create a list containing the stakeholder and their number of mentions
# The value for each session is only 1 because we only count a match
# for each session
all_orgs_count = []
for i in range(0, 27):
    session_sth_dict = defaultdict(int)
    # First extract the stakeholders
    for k, v in session_mentions_dict[i].items():
        session_sth_dict[k] += v
    # Update with the attendees
    for k, v in thai_party[i].items():
        session_sth_dict[k] += v
    for k, v in chinese_party[i].items():
        session_sth_dict[k] += v
    for k, v in lao_party[i].items():
        session_sth_dict[k] += v
    # Combine all dictionaries
    all_orgs_count.append(session_sth_dict)

#----- Network analysis
# Create an adjacency matrix, prefilled with zeros
oids = org_id.org_id
n = len(oids)
adjmat = pd.DataFrame(np.zeros((n,n)),
                      index = oids,
                      columns = oids)

for i in range(0,27):
    oid_pairs = create_sth_pairs(all_orgs_count[i])
    # Add a score of 1 to each pair in the adjacency matrix
    for pair in oid_pairs:
        adjmat.loc[pair[0], pair[1]] += 1

# Delete non-attendees/non-stakeholders
col2drop = adjmat.columns[adjmat.sum(axis=1) == 0]
adjmat.drop(col2drop, inplace=True, axis=0)
adjmat.drop(col2drop, inplace=True, axis=1)

# Plot the nodes
# Map the nodes to their respective org_id
node_map = {k: v for k, v in enumerate(adjmat.columns)}
# Create a graph object from the adjacency matrix
G = nx.from_numpy_matrix(adjmat.values)
# Create a mapping to add two attributes to a node: country and org_type
node_attrs = defaultdict(str)

# Create a set of attendees for separating attendees from stakeholders
att_set = set()
for att in attendees:
    att_set.update(att.keys())

# Create a list of nodes according to the country
th_attendees = []
th_sths = []
cn_attendees = []
cn_sths = []
lp_attendees = [] # there's no specific lao stakeholder mentioned

for i, node in enumerate(G.nodes()):
    # "i" corresponds to the node index in the graph object
    oid = node_map[i]
    if org_id[org_id['org_id'] == oid]['country'].values[0] == 'th':
        if oid in att_set:
            th_attendees.append(node)
        else:
            th_sths.append(node)
    elif org_id[org_id['org_id'] == oid]['country'].values[0] == 'cn':
        if oid in att_set:
            cn_attendees.append(node)
        else:
            cn_sths.append(node)
    else:
        lp_attendees.append(node)

pos = nx.spring_layout(G, seed=41) #seed=42
fig, ax = plt.subplots(figsize = (11,7), dpi=300)
nx.draw_networkx(G,
                 pos = pos,
                 node_size = 0,
                 width = 1, 
                 with_labels = True,
                 labels = node_map,
                 font_size=9,
                 font_color = 'red',
                 font_weight = 'semibold',
                 ax = ax)
# Solid square nodes for the thai attendees
nx.draw_networkx_nodes(G,
                       pos,
                       nodelist = th_attendees,
                       node_shape = 's')
# Light square nodes for the thai attendees
nx.draw_networkx_nodes(G,
                       pos,
                       nodelist = th_sths,
                       node_shape = 's',
                       alpha = 0.35)
# Solid circle nodes for the chinese attendees
nx.draw_networkx_nodes(G,
                       pos,
                       nodelist = cn_attendees,
                       node_shape = 'o')
# Light circle nodes for the chinese attendees
nx.draw_networkx_nodes(G,
                       pos,
                       nodelist = cn_sths,
                       node_shape = 'o',
                       alpha = 0.35)
# Solid diamond nodes for the lao attendees
nx.draw_networkx_nodes(G,
                       pos,
                       nodelist = lp_attendees,
                       node_shape = 'd')
plt.title('Spring layout')

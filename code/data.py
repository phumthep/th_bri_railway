# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 23:36:59 2021 by Phumthep
"""

from bri_util import *
import pandas as pd
import numpy as np
from collections import defaultdict


# Read all the files
stakeholder_id = pd.read_excel('stakeholder_id.xlsx', 
                               usecols = list(range(0,5)))

org_id = pd.read_excel('org_id.xlsx', 
                       usecols = list(range(0,5)))

meeting_info = pd.read_excel('meeting_info.xlsx', 
                             usecols = list(range(0,7)),
                             dtype=str) 

minutes = pd.read_excel('minutes.xlsx', 
                        usecols = list(range(1,9)))

issue_id = pd.read_excel('issue_id.xlsx', 
                         usecols = list(range(0,3)))

project_id = pd.read_excel('project_id.xlsx', 
                           usecols = list(range(0,3)))


#######------- Process the stakeholder_id
# Create a mapping matching the stakeholder_id to org_id
stakeholder_mapping = pd.Series(stakeholder_id.org_id.values,
                                index = stakeholder_id.stakeholder_id.values)\
    .to_dict()


#######------- Process the minutes dataframe
# Create a new column containing org mentions and the number of times
minutes['mentions'] = minutes['stakeholders_mentioned']\
    .apply(count_rep, args=((stakeholder_mapping), ))


#######------- Process the meeting_info
# Convert the thai_part and chinese_party to dictionaries of org counts
meeting_info['thai_party'] = meeting_info['thai_party'].apply(count_rep,
                                                              args=((stakeholder_mapping), ))
meeting_info['chinese_party'] = meeting_info['chinese_party'].apply(count_rep,
                                                              args=((stakeholder_mapping), ))
meeting_info['other'] = meeting_info['other'].apply(count_rep,
                                                    args=((stakeholder_mapping), ))

# Combine the attendees from both sides to create a new column
thai_party = meeting_info['thai_party'].values.copy()
chinese_party = meeting_info['chinese_party'].values.copy()
lao_party = meeting_info['other'].values.copy()
attendees = []
for i in range(0, 27):
    attendees.append(
        {**thai_party[i], **chinese_party[i], **lao_party[i]}
        )
meeting_info['attendees'] = pd.Series(attendees)

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 09:00:20 2021 @author: phumthep
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import permutations

def count_rep(person, mapping):
    '''Return a dictionary containing the counts of representative 
    from each organization when given a string of stakeholder ids'''
    org_counts = defaultdict(int)
    person_str = str(person)
    # Process if the string is not nan
    if person_str != 'nan':
        pids = person_str.split(',')
        # Relabel person(s) to their respective organization
        pids = [mapping[int(pid)] for pid in pids]
        # Start counting the representatives from each organization
        for pid in pids:
            org_counts[pid] += 1
    return org_counts


def get_org_name(org_id, mapping):
    '''Return the organization name from a given id'''
    return mapping[org_id]
        

def count_mentions_in_session(minutes):
    '''Return a dictionary containing org_id and their number 
    of mentions in every meeting session'''
    session_mentions = []
    df = minutes[['session', 'mentions']]
    # Create a dictionary to count mentions in every session
    for i in range(1,28):
        subset = df[df.session == i]
        total_count = defaultdict(int)
        for mention in subset.mentions:
            for k, v in mention.items():
                total_count[k] += v
        session_mentions.append(total_count)
    return session_mentions
    

def create_sth_pairs(session_sth):
    """ Return a tuple containing (org_id, org_id, 1) to represent
    the fact that they are both present in a meeting
    """
    # input session_sth is a dictionary
    oids_in_session = session_sth.keys()
    return list(permutations(oids_in_session, 2))


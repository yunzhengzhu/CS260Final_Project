import configparser
import networkx as nx
import itertools
import math
import random
import json
from tqdm import tqdm
import sys
import time
import timeit
import numpy as np
import torch
from collections import Counter
import spacy
from scipy import spatial
import sys

config = configparser.ConfigParser()
config.read("paths.cfg")


cpnet = None
cpnet_simple = None
concept2id = None
relation2id = None
id2relation = None
id2concept = None


nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])

#load concept2id, relation2id, id2relation, id2concept
def load_resources():
    global concept2id, relation2id, id2relation, id2concept
    concept2id = {}
    id2concept = {}
    with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            concept2id[w.strip()] = len(concept2id)
            id2concept[len(id2concept)] = w.strip()

    print("concept2id done")
    id2relation = {}
    relation2id = {}
    with open(config["paths"]["relation_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            id2relation[len(id2relation)] = w.strip()
            relation2id[w.strip()] = len(relation2id)

    print("relation2id done")

# turn edges into nx graph
def load_cpnet():
    global cpnet,concept2id, relation2id, id2relation, id2concept, cpnet_simple
    print("loading cpnet....")
    cpnet = nx.read_gpickle(config["paths"]["conceptnet_en_graph"])
    print("Done")

    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)

# load all concepts from dev and train concepts_nv.json
# write all oncepts into "/total_concepts.txt"
def load_total_concepts(data_path):    
    global concept2id, total_concepts_id, config
    total_concepts = []
    total_concepts_id = []
    exs = []
    for path in [data_path + "/train/concepts_nv.json", data_path + "/dev/concepts_nv.json"]:
        with open(path, 'r') as f:
            for line in f.readlines():
                line = json.loads(line)
                total_concepts.extend(line['qc'] + line['ac'])

        total_concepts = list(set(total_concepts))

    # filter total concepts not existed in concept2id (feel like useless, already filtered out in ground_concepts_simple)
    filtered_total_conncepts = []
    for x in total_concepts:
        if concept2id.get(x, False): # return False if x does not exist in concept2id (config["paths"]["concept_vocab"])
            total_concepts_id.append(concept2id[x])
            filtered_total_conncepts.append(x)

    # write filtered concepts to total concepts
    with open(data_path + "/total_concepts.txt", 'w') as f:
        for line in filtered_total_conncepts:
            f.write(str(line) + '\n')

# return list of all types of relations between two concepts
def get_edge(src_concept, tgt_concept):
    global cpnet, concept2id, relation2id, id2relation, id2concept
    try:
        rel_list = cpnet[src_concept][tgt_concept]
        return list(set([rel_list[item]["rel"] for item in rel_list]))
    except:
        return []


def find_neighbours_frequency(source_concepts, target_concepts, T, max_B=100):
    global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple, total_concepts_id
    # source concepts ids
    source = [concept2id[s_cpt] for s_cpt in source_concepts]
    start = source

    # Vts = {src: 0, ...}
    Vts = dict([(x,0) for x in start])
    Ets = {}
    total_concepts_id_set = set(total_concepts_id)

    for t in range(T):
        V = {} # number of times that a node is counted as the neighbor of a starting node
        for s in start:
            if s in cpnet_simple:
                for n in cpnet_simple[s]:
                    if n not in Vts and n in total_concepts_id_set:
                        # record frequency of neighbor node n
                        if n not in Vts: # not a starting node
                            if n not in V:
                                V[n] = 1
                            else:
                                V[n] += 1

                        # Ets[n] = {start node: relations that connect s and n}
                        if n not in Ets:
                            rels = get_edge(s, n)
                            if len(rels) > 0:
                                Ets[n] = {s: rels}
                        # add the s start node into edge-to dictionary of node n
                        else:
                            rels = get_edge(s, n)
                            if len(rels) > 0:
                                Ets[n].update({s: rels})  
                        
        # Sort the neighbor list and choose the top B most frequent
        V = list(V.items())
        count_V = sorted(V, key=lambda x: x[1], reverse=True)[:max_B]
        start = [x[0] for x in count_V if x[0] in total_concepts_id_set] #total_concepts_id = all ids
        
        # update Vts with the current top max_B start set
        Vts.update(dict([(x, t+1) for x in start]))
    
    _concepts = list(Vts.keys())
    _distances = list(Vts.values())

    # why do we need this iteration???
    concepts = []
    distances = []
    for c, d in zip(_concepts, _distances):
        concepts.append(c)
        distances.append(d)
    assert(len(concepts) == len(distances))
    
    
    triples = []
    for v, N in Ets.items():
        if v in concepts:
            for u, rels in N.items():
                # edge u -> rel -> v
                if u in concepts:
                    triples.append((u, rels, v))
    
    # target concept ids
    ts = [concept2id[t_cpt] for t_cpt in target_concepts]

    # labels := whether a concept is found in the answer concepts
    labels = []
    found_num = 0
    for c in concepts:
        if c in ts:
            found_num += 1
            labels.append(1)
        else:
            labels.append(0)
    
    res = [id2concept[x].replace("_", " ") for x in concepts]
    triples = [(id2concept[x].replace("_", " "), y, id2concept[z].replace("_", " ")) for (x,y,z) in triples]

    # concepts: concepts found in T-hop graph
    # labels: whether these concepts exist in the target
    # distances: the longest hop distance from source concepts
    # triples: all the u -> r -> v triples in the graph
    return {"concepts":res, "labels":labels, "distances":distances, "triples":triples}, found_num, len(res)

def process(input_path, output_path, T, max_B):
    data = []
    with open(input_path, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))

    examples = []
    avg_len = 0
    for ex in tqdm(data):
        target = ex['ac']
        source = ex['qc']
        
        e, _, avg_nodes = find_neighbours_frequency(source, target, T, max_B)
        avg_len += avg_nodes # number of concepts in the sub-graph
        examples.append(e)
        
    
    print('{} hops avg nodes: {}'.format(T, avg_len / len(examples)))
    
    with open(output_path, 'w') as f:
        for line in examples:
            json.dump(line ,f)
            f.write('\n')



dataset = sys.argv[1]

# 2-hop graph and top-100 most frequent neighbors 
T = 2
max_B = 100
DATA_PATH = config["paths"][dataset + "_dir"]


load_resources() # load concept2id, relation2id, id2relation, id2concept
load_cpnet()     # turn edges into nx graph

# load all concepts from dev and train concepts_nv.json
# write all oncepts into "/total_concepts.txt"
load_total_concepts(DATA_PATH)

process(DATA_PATH + "/train/concepts_nv.json", DATA_PATH + "/train/{}hops_{}_triple.json".format(T, max_B), T, max_B)
process(DATA_PATH + "/dev/concepts_nv.json", DATA_PATH + "/dev/{}hops_{}_triple.json".format(T, max_B), T, max_B)
process(DATA_PATH + "/test/concepts_nv.json", DATA_PATH + "/test/{}hops_{}_triple.json".format(T, max_B), T, max_B)
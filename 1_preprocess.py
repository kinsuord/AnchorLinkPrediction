#!usr/bin/env python
# -*- coding:utf-8 _*-

#%% Setup
import pandas as pd
import networkx as nx
import torch
import pickle
import community
from collections import defaultdict
import random
import numpy as np
from tqdm import tqdm 

target_network = ["myspace", "flickr"]

#%% Run partition
def partition(graph, resolution=1):
    part = community.best_partition(graph, resolution=resolution)
    community_number = max(part.values()) + 1
    print("community number: ", community_number, "nodesize", len(graph.nodes))
    nodes_part_list = []
    com2nodes = defaultdict(list)
    for node, com in part.items():
        com2nodes[com].append(node)

    for com, node_list in com2nodes.items():
        if len(node_list) > 15000:  
            rec_part_list = partition(nx.subgraph(graph, node_list))
            nodes_part_list += rec_part_list
        elif len(node_list) < 1000:  # 1000
            # print('community {} size {} and we ignore it'.format(com, len(node_list)))
            continue
        else:
            # print('community {} size {}'.format(com, len(node_list)))
            nodes_part_list.append(node_list)

    print('we have {} communities and each of them is larger than 1000'.format(len(nodes_part_list)))
    return nodes_part_list

for network_name in target_network:
    graph = nx.read_edgelist("./dataset/{n}/{n}.edges".format(n=network_name))
    nodes_part_list = partition(graph)
    print([len(part) for part in nodes_part_list])
    pickle.dump(nodes_part_list, open('./dataset/{n}/{n}.nodes_part_list'.format(n=network_name), 'wb'))

#%% Preproccess with shared node
for network_name in target_network:
    """
    network_1 shared_number:1056, 0:1055(含)是公共节点集合, 合并后变成了67个社区
    network_2 shared_number:1138, 0:1137(含)是公共节点集合, 合并后变成了87个社区
    """
    all_parts_name2index = defaultdict(dict)
    global_name2index = defaultdict(int)
    if network_name == 'flickr':
        shared_number = 500
    elif network_name == 'myspace':
        shared_number = 500

    """nodes_part_list: a dictionary
    {
        community_id:[node_id,node_id]
    }
    """
    nodes_part_list = pickle.load(open('./dataset/{n}/{n}.nodes_part_list'.format(n=network_name), 'rb'))
    
    g_count = 0
    for part_name, part in enumerate(nodes_part_list):
        name2index_part = defaultdict(int)
        for index, node in enumerate(part):
            name2index_part[node] = index
            if part_name == 0:
                global_name2index[node] = g_count
                g_count = g_count + 1
            elif (part_name > 0) and (index >= shared_number):
                global_name2index[node] = g_count
                g_count = g_count + 1

        pickle.dump(name2index_part, open('./dataset/{n}/{n}_{}.name2index'.format(part_name, n=network_name), 'wb'))

        all_parts_name2index[part_name] = name2index_part

    pickle.dump(all_parts_name2index, open('./dataset/{n}/{n}_all_parts.name2index'.format(n=network_name), 'wb'))
    pickle.dump(global_name2index, open('./dataset/{n}/{n}_global.name2index'.format(n=network_name), 'wb'))

#%% Gen adj matrix

for network_name in target_network:
    graph = nx.read_edgelist("./dataset/{name}/{name}.edges".format(name=network_name))
    nodes_part_list = pickle.load(open('./dataset/{n}/{n}.nodes_part_list'.format(n=network_name), 'rb'))
    
    for part_name, part in enumerate(nodes_part_list):
        name2index_part = pickle.load(open('./dataset/{n}/{n}_{}.name2index'.format(part_name, n=network_name), 'rb'))
        subG = graph.subgraph(name2index_part.keys())
        reG = nx.relabel_nodes(subG, name2index_part)
        reA = nx.adjacency_matrix(reG)
        a_tensor = torch.from_numpy(reA.A)
        torch.save(a_tensor, './dataset/{n}/{n}_{}.adj'.format(part_name, n=network_name))

#%% Sample link
negitive_sample = 5

def sample_non_neighbor_node(g, v):
    random_node = random.choice(g.nodes())
    while random_node in g.neighbors(v):
        random_node = random.choice(g.nodes())
    return random_node
    

for network_name in target_network:
    graph = nx.read_edgelist("./dataset/{name}/{name}.edges".format(name=network_name))
    nodes_part_list = pickle.load(open('./dataset/{n}/{n}.nodes_part_list'.format(n=network_name), 'rb'))

    for part_name, part in enumerate(tqdm(nodes_part_list)):
        edgelist = []
        name2index_part = pickle.load(open('./dataset/{n}/{n}_{}.name2index'.format(part_name, n=network_name), 'rb'))
        subG = graph.subgraph(name2index_part.keys())
        reG = nx.relabel_nodes(subG, name2index_part)

        link_us = []
        link_vs = []
        target = []
        degrees_weight = np.array([ reG.degree(v) for v in reG.nodes()]) ** 0.75
        node_set = set(reG.nodes())
        for edge in reG.edges:
            # add positive edges
            link_us.append(edge[0])
            link_vs.append(edge[1])
            target.append(1)

            for vertex in edge:
                non_neighbors_set = node_set - set(reG.neighbors(vertex))
                condidates = np.array(list(non_neighbors_set))
                p = degrees_weight[condidates]
                p /= p.sum()
                neg_node = np.random.choice(condidates, negitive_sample, replace=False, p=p)

                link_us += [vertex] * len(neg_node)
                link_vs += list(neg_node)
                target += [0] * len(neg_node)
            
        df = pd.DataFrame({'u': link_us,
                   'v': link_vs,
                   'target': target})
        
        file = './dataset/{n}/{n}_{}.link'.format(part_name, n=network_name)
        df.to_csv(file, index=False)

#%% cal theta
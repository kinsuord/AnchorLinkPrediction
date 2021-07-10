#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
Gen:
    '{}/{}.nodes_part_list'.format(save_dir, network_name)

    '{}/{}/{}.name2index'.format(save_dir, network_name, part_name)
    '{}/{}_all_parts.name2index'.format(save_dir, network_name)
    '{}/{}_global.name2index'.format(save_dir, network_name)

    '{}/{}/{}.adj'.format(save_dir, network_name, part_name)
    '{}/{}/{}.links'.format(save_dir, network_name, part_name)
"""

#%% Setup
import pandas as pd
import networkx as nx
import torch
import pickle
import community
from collections import defaultdict
import os
import random
import yaml
import numpy as np
from tqdm import tqdm 
import random
from sklearn.model_selection import train_test_split

param_name = "gcn"
with open("params_{}.yaml".format(param_name), "r") as f:
    params = yaml.safe_load(f)

save_dir = os.path.join("save", param_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
target_network = params['network_name']

for network_name in target_network:
    if not os.path.exists( os.path.join(save_dir, network_name)):
        os.makedirs(os.path.join(save_dir, network_name))
# #%% remove self loop in myspace
# network_name = "myspace"

# edges = []
# with open("./dataset/{name}/{name}.edges".format(name=network_name), 'r') as f:
#     for line in f.readlines():
#         u, v = line.split(" ")
#         if u != v[:-1]:
#             edges.append(line)

# with open("./dataset/{name}/{name}.edges".format(name=network_name), 'w') as f:
#     f.write(''.join(edges))

#%% Run partition
def partition(graph):
    part = community.best_partition(graph)
    community_number = max(part.values()) + 1
    print("community number: ", community_number, "nodesize", len(graph.nodes))
    nodes_part_list = []
    com2nodes = defaultdict(list)
    for node, com in part.items():
        com2nodes[com].append(node)

    for com, node_list in com2nodes.items():
        if len(node_list) > params['partition']['max']:  
            rec_part_list = partition(nx.subgraph(graph, node_list))
            nodes_part_list += rec_part_list
        elif len(node_list) < params['partition']['min']:
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
    nodes_part_list_path = '{}/{}.nodes_part_list'.format(save_dir, network_name)
    pickle.dump(nodes_part_list, open(nodes_part_list_path, 'wb'))

#%% Preproccess with shared node
for network_name in target_network:
    """
    network_1 shared_number:1056, 0:1055(含)是公共节点集合, 合并后变成了67个社区
    network_2 shared_number:1138, 0:1137(含)是公共节点集合, 合并后变成了87个社区
    """
    all_parts_name2index = defaultdict(dict)
    global_name2index = defaultdict(int)
    shared_number = params['shared_node_in_part']

    """nodes_part_list: a dictionary
    {
        community_id:[node_id,node_id]
    }
    """
    nodes_part_list_path = '{}/{}.nodes_part_list'.format(save_dir, network_name)
    nodes_part_list = pickle.load(open(nodes_part_list_path, 'rb'))
    
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

        name2index_part_path = '{}/{}/{}.name2index'.format(save_dir, network_name, part_name)
        pickle.dump(name2index_part, open(name2index_part_path, 'wb'))

        all_parts_name2index[part_name] = name2index_part

    all_parts_name2index_path = '{}/{}_all_parts.name2index'.format(save_dir, network_name)
    pickle.dump(all_parts_name2index, open(all_parts_name2index_path, 'wb'))
    global_name2index_path = '{}/{}_global.name2index'.format(save_dir, network_name)
    pickle.dump(global_name2index, open(global_name2index_path, 'wb'))

#%% Gen adj matrix

for network_name in target_network:
    graph = nx.read_edgelist("./dataset/{name}/{name}.edges".format(name=network_name))
    nodes_part_list_path = '{}/{}.nodes_part_list'.format(save_dir, network_name)
    nodes_part_list = pickle.load(open(nodes_part_list_path, 'rb'))

    for part_name, part in enumerate(nodes_part_list):
        name2index_part_path = '{}/{}/{}.name2index'.format(save_dir, network_name, part_name)
        name2index_part = pickle.load(open(name2index_part_path, 'rb'))

        subG = graph.subgraph(name2index_part.keys())
        reG = nx.relabel_nodes(subG, name2index_part)
        reA = nx.adjacency_matrix(reG)
        a_tensor = torch.from_numpy(reA.A.astype(np.float32))

        adj_tensor_path = '{}/{}/{}.adj'.format(save_dir, network_name, part_name)
        torch.save(a_tensor, adj_tensor_path)

#%% Sample link
negitive_sample = params['negitive_link_num']

def sample_non_neighbor_node(g, v):
    random_node = random.choice(g.nodes())
    while random_node in g.neighbors(v):
        random_node = random.choice(g.nodes())
    return random_node
    
for network_name in target_network:
    graph = nx.read_edgelist("./dataset/{name}/{name}.edges".format(name=network_name))
    nodes_part_list_path = '{}/{}.nodes_part_list'.format(save_dir, network_name)
    nodes_part_list = pickle.load(open(nodes_part_list_path, 'rb'))

    for part_name, part in enumerate(tqdm(nodes_part_list)):
        name2index_part_path = '{}/{}/{}.name2index'.format(save_dir, network_name, part_name)
        name2index_part = pickle.load(open(name2index_part_path, 'rb'))
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

        links_path = '{}/{}/{}.links'.format(save_dir, network_name, part_name)
        df.to_csv(links_path, header=False,index=False)

# #%% cal hyperedge theta
# n_hop_beighbor = 10

# for network_name in target_network:
#     graph = nx.read_edgelist("./dataset/{name}/{name}.edges".format(name=network_name))
#     nodes_part_list = pickle.load(open('./dataset/{n}/{n}.nodes_part_list'.format(n=network_name), 'rb'))

#     for part_name, part in enumerate(tqdm(nodes_part_list)):
#         name2index_part = pickle.load(open('./dataset/{n}/{n}_{}.name2index'.format(part_name, n=network_name), 'rb'))
#         subG = graph.subgraph(name2index_part.keys())
#         reG = nx.relabel_nodes(subG, name2index_part)
#         reA = nx.adjacency_matrix(reG)

#         H = np.ones([len(reG.nodes()), len(reG.nodes())])
#         degree_mat_inv_sqrt = np.diag(reA.A.sum(1) ** -0.5)
#         theta = degree_mat_inv_sqrt @ H @ H @ degree_mat_inv_sqrt
#         theta_tensor = torch.from_numpy(theta).to(torch.float32)
#         torch.save(theta_tensor, './dataset/{n}/{n}_{}.theta'.format(part_name, n=network_name))

#%% split anchor link

anchor_list_file = "./dataset/{}-{}.map.raw".format(target_network[0], target_network[1])

with open(anchor_list_file, 'r') as f:
    anchors = f.read().splitlines()
    anchors = [ line.split('\t', 1) for line in anchors]

with open("./dataset/{n}/{n}.nodes".format(n=target_network[0]), 'r', encoding="utf-8") as f:
    n1_nodedict = f.readlines()
    n1_nodedict = [line[:-1].split('\t', 1) for line in n1_nodedict]
    n1_nodedict = [[line[1], line[0]] for line in n1_nodedict]
    n1_nodedict = dict(n1_nodedict)

with open("./dataset/{n}/{n}.nodes".format(n=target_network[1]), 'r', encoding="utf-8") as f:
    n2_nodedict = f.readlines()
    n2_nodedict = [ line[:-1].split('\t', 1) for line in n2_nodedict]
    n2_nodedict = [[line[1], line[0]] for line in n2_nodedict]
    n2_nodedict = dict(n2_nodedict)

anchor_us = []
anchor_vs = []
for names in anchors:
    if names[0] not in n1_nodedict:
        print("n1_nodedict key not found", names[0])
        continue
    if names[1] not in n2_nodedict:
        print("n2_nodedict key not found", names[1])
        continue
    anchor_us.append(n1_nodedict[names[0]])
    anchor_vs.append(n2_nodedict[names[1]])

df = pd.DataFrame({
    target_network[0]: anchor_us,
    target_network[1]: anchor_vs,
    "value": [1] * len(anchor_us)})

train, test = train_test_split(df, test_size=0.2)

train_anchor_path = '{}/observed_anchors.positive'.format(save_dir)
train.to_csv(train_anchor_path, header=False,index=False)
test_anchor_path = '{}/test_anchors.positive'.format(save_dir)
test.to_csv(test_anchor_path, header=False,index=False)

# add negitive anchors
def random_choice(list, exclude):
    while True:
        ans = random.choice(list)
        if ans != exclude:
            return ans

nodes_1 = list(n1_nodedict.values())
nodes_2 = list(n2_nodedict.values())

def append_n_sample(df):
    n_links_1 = []
    n_links_2 = []
    for _, link in df.iterrows():
        n_links_1.append(link[0])
        n_links_2.append(random_choice(nodes_2, link[1]))
        n_links_1.append(random_choice(nodes_1, link[0]))
        n_links_2.append(link[1])
    
    df_n = pd.DataFrame({
            target_network[0]: n_links_1,
            target_network[1]: n_links_2,
            "value": [0] * len(n_links_1)})
    return df.append(df_n)

train = append_n_sample(train)
test = append_n_sample(test)
train_anchor_path = '{}/observed_anchors_p_n.df'.format(save_dir)
train.to_csv(train_anchor_path, header=False,index=False)
test_anchor_path = '{}/test_anchors_p_n.df'.format(save_dir)
test.to_csv(test_anchor_path, header=False,index=False)

#%%
for data_name in target_network:
    path_prefix = "{}/{}".format(save_dir, data_name)

    all_parts_name2index = pickle.load(open('{}_all_parts.name2index'.format(path_prefix), 'rb'))
    global_name2index = pickle.load(open('{}_global.name2index'.format(path_prefix), 'rb'))
    nodes_part_list_path = '{}.nodes_part_list'.format(path_prefix)
    nodes_part_list = pickle.load(open(nodes_part_list_path, 'rb'))
    
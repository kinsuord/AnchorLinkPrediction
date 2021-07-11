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

#%% Run partition and save nodes_part_list
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
    print([len(part) for part in nodes_part_list])
    nodes_part_list = partition(graph)
    nodes_part_list_path = '{}/{}.partition'.format(save_dir, network_name)
    pickle.dump(nodes_part_list, open(nodes_part_list_path, 'wb'))

#%% Save positive anchor
anchor_list_file = "./dataset/{}-{}.map.raw".format(target_network[0], target_network[1])

with open(anchor_list_file, 'r') as f:
    anchors = f.read().splitlines()
    anchors = [ line.split('\t', 1) for line in anchors]
print("Anchor in dataset", len(anchors))

def load_nodedict(path):
    with open(path, 'r', encoding="utf-8") as f:
        nodedict = f.readlines()
        nodedict = [line[:-1].split('\t', 1) for line in nodedict]
        nodedict = [[line[1], line[0]] for line in nodedict]
        nodedict = dict(nodedict)
    return nodedict

n1_nodedict = load_nodedict("./dataset/{n}/{n}.nodes".format(n=target_network[0]))
n2_nodedict = load_nodedict("./dataset/{n}/{n}.nodes".format(n=target_network[1]))

partition_path = '{}/{}.partition'.format(save_dir, target_network[0])
partition_1 = pickle.load(open(partition_path, 'rb'))
nodes_1 = [item for sublist in partition_1 for item in sublist]
partition_path = '{}/{}.partition'.format(save_dir, target_network[1])
partition_2 = pickle.load(open(partition_path, 'rb'))
nodes_2 = [item for sublist in partition_2 for item in sublist] 

anchor_us = []
anchor_vs = []
for names in anchors:
    if names[0] not in n1_nodedict:
        print("anchor not found in nodes_1", names[0])
        continue
    if names[1] not in n2_nodedict:
        print("anchor not found in nodes_2", names[1])
        continue

    if n1_nodedict[names[0]] not in nodes_1:
        continue
    if n2_nodedict[names[1]] not in nodes_2:
        continue

    anchor_us.append(n1_nodedict[names[0]])
    anchor_vs.append(n2_nodedict[names[1]])

df = pd.DataFrame({
    target_network[0]: anchor_us,
    target_network[1]: anchor_vs,
    "value": [1] * len(anchor_us)})
print("Only", df.shape[0], "in the partition")

train, test = train_test_split(df, test_size=0.2)

train_anchor_path = '{}/observed_anchors.positive'.format(save_dir)
train.to_csv(train_anchor_path, header=False,index=False)
test_anchor_path = '{}/test_anchors.positive'.format(save_dir)
test.to_csv(test_anchor_path, header=False,index=False)

# Sample negitive anchors
def random_choice(list, exclude):
    while True:
        ans = random.choice(list)
        if ans != exclude:
            return ans

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

#%% Preprocess node partition list
train_anchor_path = '{}/observed_anchors.positive'.format(save_dir)
train = pd.read_csv(train_anchor_path, header=None)
test_anchor_path = '{}/test_anchors.positive'.format(save_dir)
test = pd.read_csv(test_anchor_path, header=None)

df = train.append(test)

for data_index, data_name in enumerate(target_network):
    path_prefix = "{}/{}".format(save_dir, data_name)
    partition_path = '{}.partition'.format(path_prefix)
    partition_list = pickle.load(open(partition_path, 'rb'))
    partition_list = np.array(partition_list)

    node2part = {}
    for i in range(len(partition_list)):
        node2part.update(dict.fromkeys(partition_list[i], i))

    count_anchor = np.zeros(len(partition_list))
    for _, link in df.iterrows():
        count_anchor[node2part[str(link[data_index])]] += 1

    print(len(np.where(count_anchor>0.0)[0]), "parts has anchors, total:", len(partition_list))
    clear_partition = partition_list[count_anchor>0.0]

    # add shared part
    nodes_part_with_shared = []
    for i, part in enumerate(clear_partition):
        if i == 0:
            nodes_part_with_shared.append(part)
        else:
            nodes_part_with_shared.append( nodes_part_list[0] + part)

    nodes_part_list_path = '{}/{}.nodes_part_list'.format(save_dir, network_name)
    pickle.dump(nodes_part_with_shared, open(nodes_part_list_path, 'wb'))

#%% Preproccess with shared node
for network_name in target_network:

    all_parts_name2index = defaultdict(dict)
    global_name2index = defaultdict(int)

    """nodes_part_list: a dictionary
    {
        community_id:[node_id,node_id]
    }
    """
    nodes_part_list_path = '{}/{}.nodes_part_list'.format(save_dir, network_name)
    nodes_part_list = pickle.load(open(nodes_part_list_path, 'rb'))

    shared_number = len(nodes_part_list[0])
    
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

    for part_name, part in enumerate(tqdm(nodes_part_list)):
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

# #%% split anchor link

# anchor_list_file = "./dataset/{}-{}.map.raw".format(target_network[0], target_network[1])

# with open(anchor_list_file, 'r') as f:
#     anchors = f.read().splitlines()
#     anchors = [ line.split('\t', 1) for line in anchors]

# with open("./dataset/{n}/{n}.nodes".format(n=target_network[0]), 'r', encoding="utf-8") as f:
#     n1_nodedict = f.readlines()
#     n1_nodedict = [line[:-1].split('\t', 1) for line in n1_nodedict]
#     n1_nodedict = [[line[1], line[0]] for line in n1_nodedict]
#     n1_nodedict = dict(n1_nodedict)

# with open("./dataset/{n}/{n}.nodes".format(n=target_network[1]), 'r', encoding="utf-8") as f:
#     n2_nodedict = f.readlines()
#     n2_nodedict = [ line[:-1].split('\t', 1) for line in n2_nodedict]
#     n2_nodedict = [[line[1], line[0]] for line in n2_nodedict]
#     n2_nodedict = dict(n2_nodedict)

# global_name2index_path = '{}/{}_global.name2index'.format(save_dir, target_network[0])
# global_name2index_1 = pickle.load(open(global_name2index_path, 'rb'))
# global_name2index_path = '{}/{}_global.name2index'.format(save_dir, target_network[1])
# global_name2index_2 = pickle.load(open(global_name2index_path, 'rb'))

# anchor_us = []
# anchor_vs = []
# for names in anchors:
#     if names[0] not in n1_nodedict:
#         print("n1_nodedict key not found", names[0])
#         continue
#     if names[1] not in n2_nodedict:
#         print("n2_nodedict key not found", names[1])
#         continue

#     if n1_nodedict[names[0]] not in global_name2index_1:
#         # print("n1 not found", n1_nodedict[names[0]])
#         continue
#     if n2_nodedict[names[1]] not in global_name2index_2:
#         # print("n2 not found", n2_nodedict[names[1]])
#         continue

#     anchor_us.append(global_name2index_1[n1_nodedict[names[0]]])
#     anchor_vs.append(global_name2index_2[n2_nodedict[names[1]]])

# df = pd.DataFrame({
#     target_network[0]: anchor_us,
#     target_network[1]: anchor_vs,
#     "value": [1] * len(anchor_us)})

# train, test = train_test_split(df, test_size=0.2)

# train_anchor_path = '{}/observed_anchors.positive'.format(save_dir)
# train.to_csv(train_anchor_path, header=False,index=False)
# test_anchor_path = '{}/test_anchors.positive'.format(save_dir)
# test.to_csv(test_anchor_path, header=False,index=False)

# # add negitive anchors
# def random_choice(list, exclude):
#     while True:
#         ans = random.choice(list)
#         if ans != exclude:
#             return ans

# nodes_1 = list(global_name2index_1.values())
# nodes_2 = list(global_name2index_2.values())

# def append_n_sample(df):
#     n_links_1 = []
#     n_links_2 = []
#     for _, link in df.iterrows():
#         n_links_1.append(link[0])
#         n_links_2.append(random_choice(nodes_2, link[1]))
#         n_links_1.append(random_choice(nodes_1, link[0]))
#         n_links_2.append(link[1])
    
#     df_n = pd.DataFrame({
#             target_network[0]: n_links_1,
#             target_network[1]: n_links_2,
#             "value": [0] * len(n_links_1)})
#     return df.append(df_n)

# train = append_n_sample(train)
# test = append_n_sample(test)
# train_anchor_path = '{}/observed_anchors_p_n.df'.format(save_dir)
# train.to_csv(train_anchor_path, header=False,index=False)
# test_anchor_path = '{}/test_anchors_p_n.df'.format(save_dir)
# test.to_csv(test_anchor_path, header=False,index=False)

# #%% TEST
# train_anchor_path = '{}/observed_anchors.positive'.format(save_dir)
# train = pd.read_csv(train_anchor_path, header=None)
# test_anchor_path = '{}/test_anchors.positive'.format(save_dir)
# test = pd.read_csv(test_anchor_path, header=None)

# df = train.append(test)

# # for data_name in target_network:
# data_name = target_network[0]
# path_prefix = "{}/{}".format(save_dir, data_name)

# all_parts_name2index = pickle.load(open('{}_all_parts.name2index'.format(path_prefix), 'rb'))
# global_name2index = pickle.load(open('{}_global.name2index'.format(path_prefix), 'rb'))
# nodes_part_list_path = '{}.nodes_part_list'.format(path_prefix)
# nodes_part_list = pickle.load(open(nodes_part_list_path, 'rb'))

# node2part = {}
# for i in range(1, len(nodes_part_list)):
#     global_index = [ global_name2index[node] for node in nodes_part_list[i]]
#     node2part.update(dict.fromkeys(global_index, i))

# count_anchor = np.zeros(len(nodes_part_list))
# for _, link in df.iterrows():
#     count_anchor[node2part[link[0]]] += 1

# print(len(np.where(count_anchor==0.0)[0]))
# print(len(np.where(count_anchor>0.0)[0]))
#     # break
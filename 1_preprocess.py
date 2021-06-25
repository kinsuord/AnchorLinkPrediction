#!usr/bin/env python
# -*- coding:utf-8 _*-
import pandas as pd
import networkx as nx
import pickle
import pdb
import community
from collections import defaultdict

def partition(graph, resolution=1):
    part = community.best_partition(graph, resolution=resolution)
    community_number = max(part.values()) + 1
    print("community number: ", community_number)
    nodes_part_list = []
    com2nodes = defaultdict(list)
    for node, com in part.items():
        com2nodes[com].append(node)

    for com, node_list in com2nodes.items():
        if len(node_list) < 500:  # 500
            # print('community {} size {} and we ignore it'.format(com, len(node_list)))
            continue
        else:
            print('community {} size {}'.format(com, len(node_list)))
            nodes_part_list.append(node_list)

    print('we have {} communities and each of them is larger than 500'.format(len(nodes_part_list)))
    return nodes_part_list

target_network = ["flickr", "myspace"]

# for network_name in target_network:
#     graph = nx.read_edgelist("./dataset/{name}/{name}.edges".format(name=network_name))
#     nodes_part_list = partition(graph)
#     pickle.dump(nodes_part_list, open('./dataset/{name}/{}.nodes_part_list'.format(network_name), 'wb'))

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

    # pdb.set_trace()
    
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
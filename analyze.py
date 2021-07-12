#%%
# import pandas as pd
import networkx as nx
# import torch
import pickle
# import community
# from collections import defaultdict
import os
# import random
import yaml
from networkx.algorithms import distance_measures
# import numpy as np
from tqdm import tqdm 
# import random
# from sklearn.model_selection import train_test_split

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

#%% gen sub graph

dia_nets = []
for network_name in target_network:
    graph = nx.read_edgelist("./dataset/{name}/{name}.edges".format(name=network_name))
    nodes_part_list_path = '{}/{}.partition'.format(save_dir, network_name)
    nodes_part_list = pickle.load(open(nodes_part_list_path, 'rb'))

    dia = []
    for part_name, part in enumerate(tqdm(nodes_part_list)):
        subG = graph.subgraph(part)
        # reG = nx.relabel_nodes(subG, name2index_part)
        # nx.write_gpickle(subG, '{}/{}/{}.graph'.format(save_dir, network_name, part_name))
        try:
            diameter = distance_measures.diameter(reG)
            dia.append(diameter)
        except:
            dia.append(0)
    dia_nets.append(dia)


# #%%
# dia_nets = []
# for network_name in target_network:
#     nodes_part_list_path = '{}/{}.nodes_part_list'.format(save_dir, network_name)
#     nodes_part_list = pickle.load(open(nodes_part_list_path, 'rb'))

#     dia = []
#     for part_name, part in enumerate(tqdm(nodes_part_list)):
#         if part_name == 0:
#             continue
#         reG = nx.read_gpickle('{}/{}/{}.graph'.format(save_dir, network_name, part_name))
#         try:
#             diameter = distance_measures.diameter(reG)
#             dia.append(diameter)
#         except:
#             dia.append(0)

#     dia_nets.append(dia)

# %% draw diameter
import matplotlib.pyplot as plt

plt.hist(dia_nets[0], density=True)  # density=False would make counts
plt.ylabel('Density')
plt.xlabel('Diameter')
plt.title(target_network[0])
plt.show()

plt.hist(dia_nets[1], density=True)  # density=False would make counts
plt.ylabel('Density')
plt.xlabel('Diameter')
plt.title(target_network[1])
plt.show()
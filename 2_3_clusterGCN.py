import math
import yaml
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
torch.backends.cudnn.benchmark = True
print(device)


if __name__ == "__main__":

    """
    './{data_name}_{part_name}.adj' adj matrix for each partition of the dataset.
    './{data_name}_{part_name}.theta' calculated from hypergraph incident matirx(pyroch tensor)
    for a big graph, we make graph partition, each  partition is a part_name
    """
    param_name = "cluster_gcn"
    with open("params_{}.yaml".format(param_name), "r") as f:
        params = yaml.safe_load(f)
    save_dir = os.path.join("save", param_name)

    target_network = params['network_name']
    if params['embedding']['name'] == "cluster_gcn":
        pass
    else:
        raise ValueError("Unknown network name")

    epoches = params['embedding']['epoch']
    for data_name in target_network:
        path_prefix = "{}/{}".format(save_dir, data_name)

        all_parts_name2index = pickle.load(open('{}_all_parts.name2index'.format(path_prefix), 'rb'))
        part_number = len(all_parts_name2index.keys())
        for part_name in tqdm(range(part_number)):
            adj = torch.load('{}/{}.adj'.format(path_prefix, part_name)).to(device)
            print(adj.size())
            links_pd = pd.read_csv('{}/{}.links'.format(path_prefix, part_name), header=None)
            
            links = torch.from_numpy(np.array(links_pd[[0, 1]]))
            links_target = torch.from_numpy(np.array(links_pd[2])).view(-1).to(device)
            positive_links = links[links_target == 1]
            negtive_links = links[links_target == 0]

            """
            './{data_name}_{part_name}.theta' calculated from hypergraph incident matirx(pyroch tensor)
            for a big graph, we make graph partition, each  partition is a part_name
            """
            # theta_path = '{}_{}.theta'.format(path_prefix, part_name)
            # theta = torch.load(theta_path).to(device)
            if params['embedding']['name'] == "gcn_only":
                model = Machine(node_number=adj.shape[0],
                                dropout=0.0001,
                                d_1=200, d_2=0, d_3=0,
                                ini_emb_mode='par').to(device)
            else:
                raise Exception('Unknown network name')

            optimizer = torch.optim.Adam(model.parameters(), lr=params['embedding']['lr'])

            model.train()

            for epoch in range(epoches):
                optimizer.zero_grad()

                embeddings = model(adj=adj)
                pos_ran_idxs = torch.randint(0, len(positive_links), (10000,), device=device)
                neg_ran_idxs = torch.randint(0, len(negtive_links), (100000,), device=device)
                loss = model.embedding_loss(embeddings, positive_links[pos_ran_idxs], negtive_links[neg_ran_idxs])
                # loss = model.embedding_loss(embeddings, positive_links[:10000], negtive_links[:100000])
                # loss = model.embedding_loss(embeddings, positive_links, negtive_links)
                
                if epoch % 200 == 0:
                    print("{} | part {}/{} | epoch {}/{} | loss: {:.4f}".format(data_name, part_name, part_number,
                                                                               epoch, epoches, loss.item()))
                loss.backward()
                optimizer.step()
            
            model.save_embeddings('{}/{}.embedding'.format(path_prefix, part_name))

            left_p = embeddings[positive_links[pos_ran_idxs][:, 0]]
            right_p = embeddings[positive_links[pos_ran_idxs][:, 1]]
            dots_p = torch.sum(torch.mul(left_p, right_p), dim=1)
            positive_loss = torch.mean(-1.0 * F.logsigmoid(dots_p))
            left_n = embeddings[negtive_links[neg_ran_idxs][:, 0]]
            right_n = embeddings[negtive_links[neg_ran_idxs][:, 1]]
            dots_n = torch.sum(torch.mul(left_n, right_n), dim=1)
            negtive_loss = torch.mean(-1.0 * torch.log(1.01 - torch.sigmoid(dots_n)))
            print("dots_p", dots_p.mean().item(), "dots_n",dots_n.mean().item())

            del adj, model, loss, embeddings, optimizer, links_target
            torch.cuda.empty_cache()

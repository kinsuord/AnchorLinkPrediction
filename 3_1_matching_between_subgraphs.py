import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import yaml
from tqdm import tqdm 
import pickle

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(device)

class Matching(nn.Module):
    def __init__(self, d_embedding_before):
        super(Matching, self).__init__()
        self.fc1 = nn.Linear(d_embedding_before, d_embedding_before)
        self.fc2 = nn.Linear(d_embedding_before, d_embedding_before)

    def forward(self, embedding_1, embedding_2, observed_anchors_p):

        self.embedding_1_after = embedding_1
        # self.embedding_1_after = F.dropout(self.embedding_1_after, 0.5) # matching 的时候千万不要加dropout！
        # self.embedding_1_after = self.fc2(self.embedding_1_after)
        self.embedding_2_after = self.fc1(embedding_2)
        # print(self.embedding_1_after)
        # x_1_p = self.embedding_1_after[observed_anchors_p[:, 0]]
        # x_2_p = self.embedding_2_after[observed_anchors_p[:, 1]]
        x_1_p = self.embedding_1_after.index_select(0, observed_anchors_p[:, 0]) 
        x_2_p = self.embedding_2_after.index_select(0, observed_anchors_p[:, 1]) 

        dots_p = torch.sum(torch.mul(x_1_p, x_2_p), dim=1)
        positive_loss = torch.mean(-1.0 * F.logsigmoid(dots_p))
        return positive_loss

        #dis_p = torch.mean(F.pairwise_distance(x_1_p, x_2_p))
        #return dis_p

    def save_embeddings(self, embedding_1_path='network_1', embedding_2_path='network_2'):
        torch.save(self.embedding_1_after, embedding_1_path)
        torch.save(self.embedding_2_after, embedding_2_path)


if __name__ == "__main__":
    param_name = "gcn"
    with open("params_{}.yaml".format(param_name), "r") as f:
        params = yaml.safe_load(f)
    save_dir = os.path.join("save", param_name)

    epoches = params['match_part']['epoch']

    print('网络内matching...')
    for data_name in params['network_name']:
        path_prefix = "{}/{}".format(save_dir, data_name)

        nodes_part_list_path = '{}.nodes_part_list'.format(path_prefix)
        nodes_part_list = pickle.load(open(nodes_part_list_path, 'rb'))
        shared_number = len(nodes_part_list[0])
        
        anchors_list = [[i, i] for i in range(shared_number)]
        anchors_p = torch.from_numpy(np.array(anchors_list)).to(device)  # left:0, right: others

        all_parts_name2index = pickle.load(open('{}_all_parts.name2index'.format(path_prefix), 'rb'))
        part_number = len(all_parts_name2index.keys())
        # print(part_number)
        # for mode in ['mine']:

        for part_name in tqdm(range(1, part_number)):
            embedding_1 = torch.load('{}/0.embedding'.format(path_prefix))  # .cpu()  # others
            embedding_2 = torch.load('{}/{}.embedding'.format(path_prefix, part_name))  # .cpu()  # 0

            model = Matching(embedding_1.shape[1]).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=params['match_part']['lr'], weight_decay=0.00005)

            model.train()
            for epoch in range(epoches):
                optimizer.zero_grad()

                train_loss = model(embedding_1, embedding_2, anchors_p)
                if epoch % 500 == 0:
                    print("{}| part: {}/{}->0 | {}/{} | "
                            "train_loss: {:.8f}".format(data_name, part_name, part_number, epoch, epoches,
                                                        train_loss.item()))

                train_loss.backward()
                optimizer.step()

            model.save_embeddings(
                embedding_1_path='{}/0_match_part.embedding'.format(path_prefix), 
                embedding_2_path='{}/{}_match_part.embedding'.format(path_prefix, part_name))

        # all_parts_name2index
        # {
        #     'part1': {
        #         1: 0,
        #         2: 1
        #     },
        #     'part2': {
        #         3: 0,
        #         4: 1
        #     }
        # }
        print('embedding合并...')

        embedding_list = []  #
        for part_name in range(part_number):
            emb = torch.load('{}/{}_match_part.embedding'.format(path_prefix, part_name))
            if part_name > 0:
                emb = emb[shared_number:, :]

            embedding_list.append(emb)
        embedding_global = torch.cat(embedding_list, dim=0)
        torch.save(embedding_global, '{}_global_part.embedding'.format(path_prefix))

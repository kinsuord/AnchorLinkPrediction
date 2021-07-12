import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import yaml
import pickle

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# device = "cpu"

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
        # import pdb; pdb.set_trace()
        x_1_p = self.embedding_1_after.index_select(0, observed_anchors_p[:, 0]) 
        x_2_p = self.embedding_2_after.index_select(0, observed_anchors_p[:, 1])

        # dots_p = torch.sum(torch.mul(x_1_p, x_2_p), dim=1)
        # positive_loss = torch.mean(-1.0 * F.logsigmoid(dots_p))
        # return positive_loss
        dis_p = torch.mean(F.pairwise_distance(x_1_p, x_2_p))
        return dis_p

    def save_embeddings(self, embedding_1_path='network_1', embedding_2_path='network_2'):
        torch.save(self.embedding_1_after, embedding_1_path)
        torch.save(self.embedding_2_after, embedding_2_path)   

    # def save_embeddings(self, mode='gcn_only', embedding_1_name='network_1', embedding_2_name='network_2'):
    #     torch.save(self.embedding_1_after, './' + embedding_1_name + '_after_' + mode + '.embeddings')
    #     torch.save(self.embedding_2_after, './' + embedding_2_name + '_after_' + mode + '.embeddings')


# def get_a

if __name__ == "__main__":
    param_name = "gcn"
    with open("params_{}.yaml".format(param_name), "r") as f:
        params = yaml.safe_load(f)
    save_dir = os.path.join("save", param_name)
    data_names = params['network_name']

    epoches_2 = params['match_network']['epoch']

    print('网络间matching...')

    path1_prefix = "{}/{}".format(save_dir, data_names[0])
    path2_prefix = "{}/{}".format(save_dir, data_names[1])
    embedding_1 = torch.load('{}_global_part.embedding'.format(path1_prefix)).to(device)  # .cpu() # others
    embedding_2 = torch.load('{}_global_part.embedding'.format(path2_prefix)).to(device)  # .cpu() # 0

    global_name2index_path = '{}_global.name2index'.format(path1_prefix)
    global_name2index_1 = pickle.load(open(global_name2index_path, 'rb'))
    global_name2index_path = '{}_global.name2index'.format(path2_prefix)
    global_name2index_2 = pickle.load(open(global_name2index_path, 'rb'))

    observed_anchors_pd = pd.read_csv('{}/observed_anchors.positive'.format(save_dir), header=None)  # index form, [i,i 1]
    test_anchors_pd = pd.read_csv('{}/test_anchors.positive'.format(save_dir), header=None)

    observed_anchors = []
    for _, anchor in observed_anchors_pd.iterrows():
        observed_anchors.append([global_name2index_1[str(anchor[0])], global_name2index_2[str(anchor[1])]])
    observed_anchors_p = torch.from_numpy(np.array(observed_anchors)).to(device)

    test_anchors = []
    for _, anchor in test_anchors_pd.iterrows():
        test_anchors.append([global_name2index_1[str(anchor[0])], global_name2index_2[str(anchor[1])]])
    test_anchors_p = torch.from_numpy(np.array(test_anchors)).to(device)

    # import pdb; pdb.set_trace()
    model = Matching(embedding_1.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['match_network']['lr'], weight_decay=0.00005)

    model.train()
    for epoch in range(epoches_2):
        optimizer.zero_grad()
        train_loss = model(embedding_1, embedding_2, observed_anchors_p)
        test_loss = model(embedding_1, embedding_2, test_anchors_p)

        if epoch % 200 == 0:
            print("{}/{} | "
                    "train_loss: {:.8f} | "
                    "test_loss: {:.8f}".format(epoch, epoches_2,
                                            train_loss.item(),
                                            test_loss.item()))

        train_loss.backward()
        optimizer.step()

    model.save_embeddings('{}_network_match.embedding'.format(path1_prefix),
                        '{}_network_match.embedding'.format(path2_prefix))

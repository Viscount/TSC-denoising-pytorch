#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.utils.data as data
import torch.nn.functional as F
import dgl
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable
import torch.optim as optim
import torch.autograd as autograd
from torch.optim.lr_scheduler import StepLR
import util.validation as valid_util
from tensorboardX import SummaryWriter
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # 公式 (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # 公式 (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # 公式 (2) 所需，边上的用户定义函数
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # 公式 (3), (4)所需，传递消息用的用户定义函数
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # 公式 (3), (4)所需, 归约用的用户定义函数
        # 公式 (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # 公式 (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # 公式 (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # 公式 (2)
        self.g.apply_edges(self.edge_attention)
        # 公式 (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # 对输出特征维度（第1维）做拼接
            return torch.cat(head_outs, dim=1)
        else:
            # 用求平均整合多头结果
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # 注意输入的维度是 hidden_dim * num_heads 因为多头的结果都被拼接在了
        # 一起。 此外输出层只有一个头。
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

        self.fc1 = nn.Linear(out_dim, 256, bias=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.fc3 = nn.Linear(128, 2, bias=True)

    def forward(self, sentence, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)

        h[0] = torch.zeros(h[0].shape[0])
        sent_emd = h[sentence]
        sent_emd = torch.sum(sent_emd, dim=1)

        sent_emd = F.dropout(sent_emd, p=0.5, training=self.training)
        h1 = F.relu(self.fc1(sent_emd))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(self.fc2(h1))
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h3 = self.fc3(h2)
        return h3


def build_graph(features, edges):
    g = dgl.DGLGraph()
    g.add_nodes(features.shape[0])
    src, dst, value = tuple(zip(*edges))
    g.add_edges(src, dst)
    g.add_edges(dst, src)
    for i in range(features.shape[0]):
        g.add_edge(i, i)
    g.ndata['feat'] = torch.FloatTensor(features)
    return g


def train(season_id, dm_train_set, dm_test_set, features, edges):
    EMBEDDING_DIM = 200
    batch_size = 128
    epoch_num = 200
    max_acc = 0
    max_v_acc = 0
    model_save_path = './tmp/model_save/gcn_context.model'

    dm_dataloader = data.DataLoader(
        dataset=dm_train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8
    )

    dm_test_dataloader = data.DataLoader(
        dataset=dm_test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8
    )

    graph = build_graph(features, edges)
    features = torch.FloatTensor(features)
    features = features.to(device)

    model = GAT(graph, EMBEDDING_DIM, 25, 200, 8)
    print(model)
    model.to(device)

    if torch.cuda.is_available():
        print("CUDA : On")
    else:
        print("CUDA : Off")

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    logging = True
    if logging:
        writer = SummaryWriter()
        log_name = 'gat'

    for epoch in tqdm(range(epoch_num)):
        print()
        model.train(mode=True)
        # scheduler.step()
        for batch_idx, sample_dict in enumerate(dm_dataloader):
            sentence = torch.LongTensor(sample_dict['sentence'])
            label = torch.LongTensor(sample_dict['label'])

            sentence = sentence.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            pred = model(sentence, features)
            cross_entropy = nn.CrossEntropyLoss()
            loss = cross_entropy(pred, label)
            if batch_idx % 10 == 0:
                accuracy = valid_util.running_accuracy(pred, label)
                print('epoch: %d batch %d : loss: %4.6f accuracy: %4.6f' % (epoch, batch_idx, loss.item(), accuracy))
                if logging:
                    writer.add_scalar(log_name + '_data/loss', loss.item(), epoch * 10 + batch_idx // 10)
            loss.backward()
            optimizer.step()

        model.eval()
        accuracy = valid_util.validate(model, dm_test_set, dm_test_dataloader, mode='output', type='graph',
                                       features=features, g=graph)
        if accuracy > max_acc:
            max_acc = accuracy

        if logging:
            result_dict = valid_util.validate(model, dm_test_set, dm_test_dataloader, mode='report',
                                              type='graph', features=features, g=graph)
            writer.add_scalars(log_name + '_data/0-PRF', {
                '0-Precision': result_dict['0']['precision'],
                '0-Recall': result_dict['0']['recall'],
                '0-F1-score': result_dict['0']['f1-score']
            }, epoch)
            writer.add_scalars(log_name + '_data/1-PRF', {
                '1-Precision': result_dict['1']['precision'],
                '1-Recall': result_dict['1']['recall'],
                '1-F1-score': result_dict['1']['f1-score']
            }, epoch)
            writer.add_scalars(log_name + '_data/accuracy', {
                'accuracy': result_dict['accuracy'],
                'max_accuracy': max_acc
            }, epoch)

    if logging:
        writer.close()
    print("Max Accuracy: %4.6f" % max_acc)
    return

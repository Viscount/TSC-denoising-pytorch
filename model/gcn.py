#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.module import Module
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torch_geometric.data import Data
import torch_geometric.utils as gutil
from torch_geometric.nn import GCNConv

import util.validation as valid_util
from tensorboardX import SummaryWriter
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GCN(Module):
    def __init__(self, g, in_feats, hidden_size, dropout):
        super(GCN, self).__init__()
        self.embedding_dim = in_feats
        self.graph = g
        self.embedding = nn.Embedding(g.num_nodes, in_feats, padding_idx=0)
        self.conv1 = GCNConv(in_feats, hidden_size)
        self.conv2 = GCNConv(hidden_size, in_feats)
        self.dropout = dropout

        self.fc1 = nn.Linear(in_feats, 256, bias=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.fc3 = nn.Linear(128, 2, bias=True)

    def forward(self, sentence):
        # self.graph.x = self.embedding.weight
        x, edge_index = self.graph.x, self.graph.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        sent_emd = x[sentence]
        sent_emd = torch.sum(sent_emd, dim=1)

        sent_emd = F.dropout(sent_emd, p=0.5, training=self.training)
        h1 = F.relu(self.fc1(sent_emd))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(self.fc2(h1))
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h3 = self.fc3(h2)
        return h3

    def init_emb(self, pre_train_weight):
        init_range = 1 / self.embedding_dim
        if pre_train_weight.shape == self.embedding.weight.data.shape:
            pre_train_weight[1, :] = torch.FloatTensor(np.random.uniform(-init_range, init_range, pre_train_weight.shape[1]))
            self.embedding.weight.data = pre_train_weight
        else:
            print('Weight data shape mismatch, using default init.')
        return


def build_graph(features, edges):
    x = torch.FloatTensor(features)
    src, dst, value = tuple(zip(*edges))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index = gutil.to_undirected(edge_index, features.shape[0])
    edge_index = gutil.add_self_loops(edge_index, features.shape[0])
    g = Data(x=x, edge_index=edge_index)
    return g


def train(season_id, dm_train_set, dm_test_set, features, edges):

    EMBEDDING_DIM = 200
    batch_size = 128
    epoch_num = 300
    max_acc = 0
    max_v_acc = 0
    model_save_path = './tmp/model_save/gcn.model'

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
    graph = graph.to(device)

    model = GCN(graph, EMBEDDING_DIM, 256, dropout=0.5)
    # model.init_emb(features)
    print(model)
    model.to(device)

    if torch.cuda.is_available():
        print("CUDA : On")
    else:
        print("CUDA : Off")

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    logging = False
    if logging:
        writer = SummaryWriter()
        log_name = 'gcn'

    for epoch in tqdm(range(epoch_num)):
        model.train(mode=True)
        # scheduler.step()
        for batch_idx, sample_dict in enumerate(dm_dataloader):
            sentence = torch.LongTensor(sample_dict['sentence'])
            label = torch.LongTensor(sample_dict['label'])

            sentence = sentence.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            pred = model.forward(sentence)
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
        accuracy = valid_util.validate(model, dm_test_set, dm_test_dataloader, mode='output', type='normal')
        if accuracy > max_acc:
            max_acc = accuracy

        if logging:
            result_dict = valid_util.validate(model, dm_test_set, dm_test_dataloader, mode='report', type='normal')
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

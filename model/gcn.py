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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gcn_message(edges):
    return {'msg': edges.src['h']}


def gcn_reduce(nodes):
    return {'h': torch.mean(nodes.mailbox['msg'], dim=1)}


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):
        # g is the graph and the inputs is the input node features
        # first set the node features
        g.ndata['h'] = inputs
        # trigger message passing on all edges
        g.send(g.edges(), gcn_message)
        # trigger aggregation at all nodes
        g.recv(g.nodes(), gcn_reduce)
        # get the result node features
        h = g.ndata.pop('h')
        # perform linear transformation
        return self.linear(h)


class GCN(Module):
    def __init__(self, in_feats, hidden_size, num_class, dropout):
        super(GCN, self).__init__()

        self.gc1 = GCNLayer(in_feats, hidden_size)
        self.gc2 = GCNLayer(hidden_size, num_class)
        self.dropout = dropout

        self.fc1 = nn.Linear(num_class, 256, bias=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.fc3 = nn.Linear(128, 2, bias=True)

    def forward(self, sentence, g, inputs):
        x = F.relu(self.gc1(g, inputs))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gc2(g, x)

        x[0] = torch.zeros(x[0].shape[0])
        sent_emd = x[sentence]
        sent_emd = torch.sum(sent_emd, dim=1)
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
    g.ndata['feat'] = torch.FloatTensor(features)
    return g


def train(season_id, dm_train_set, dm_test_set, features, edges):

    EMBEDDING_DIM = 200
    batch_size = 128
    epoch_num = 100
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

    model = GCN(EMBEDDING_DIM, 256, 200, dropout=0.5)
    print(model)
    model.to(device)

    if torch.cuda.is_available():
        print("CUDA : On")
    else:
        print("CUDA : Off")

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    logging = True
    if logging:
        writer = SummaryWriter()
        log_name = 'gcn'

    graph = build_graph(features, edges)
    features = torch.FloatTensor(features)
    features = features.to(device)

    for epoch in range(epoch_num):

        model.train(mode=True)
        scheduler.step()
        for batch_idx, sample_dict in enumerate(dm_dataloader):
            sentence = torch.LongTensor(sample_dict['sentence'])
            label = torch.LongTensor(sample_dict['label'])

            sentence = sentence.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            pred = model(sentence, graph, features)
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
            result_dict = valid_util.validate(model, dm_test_set, dm_test_dataloader, mode='report', type='graph',
                                       features=features, g=graph)
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

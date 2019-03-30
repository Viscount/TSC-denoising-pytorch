#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import os
import pickle
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch.nn.modules.module import Module
from torch.nn import Parameter
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torch_geometric.data import Data
import torch_geometric.utils as gutil
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

import networkx as nx
import util.validation as valid_util
from tensorboardX import SummaryWriter
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WeightLayer(nn.Module):
    def __init__(self, in_dim, bias=True):
        super(WeightLayer, self).__init__()
        self.weight = Parameter(torch.Tensor(in_dim))
        if bias:
            self.bias = Parameter(torch.Tensor(in_dim))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / self.weight.shape[0]
        init.uniform_(self.weight, 0, bound)
        if self.bias is not None:
            bound = 1 / math.sqrt(self.weight.shape[0])
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        weight = self.weight.unsqueeze(0)
        weight = weight.unsqueeze(2)
        if self.bias is not None:
            bias = self.weight.unsqueeze(0)
            bias = bias.unsqueeze(2)
            return input * weight + bias
        else:
            return input * weight


class SimpleGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super(SimpleGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index):
        """"""
        x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x = torch.add(x, self.bias)
        return self.propagate('mean', edge_index, x=x)


class GCNContext(Module):
    def __init__(self, graph, embedding_dim, hidden_size, context_mode='concat', cut_off=49):
        super(GCNContext, self).__init__()
        self.context_mode = context_mode
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(graph.num_nodes, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(cut_off+1, embedding_dim)
        self.context_fc = WeightLayer(50, bias=False)
        # self.conv1 = SimpleGCNConv(embedding_dim, hidden_size)
        # self.conv2 = SimpleGCNConv(hidden_size, embedding_dim)
        self.conv1 = SAGEConv(embedding_dim, hidden_size)
        self.conv2 = SAGEConv(hidden_size, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim * 2, momentum=0.5)

        self.fc1 = nn.Linear(embedding_dim * 2, 512, bias=True)
        self.fc2 = nn.Linear(512, 2, bias=True)

    def forward(self, sentence, context, **extra_input):
        self.graph.x = self.embedding.weight
        x, edge_index = self.graph.x, self.graph.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x += self.embedding.weight

        sent_emd = x[sentence]
        sent_emd = torch.sum(sent_emd, dim=1)

        if self.context_mode == 'concat':
            cont_emd = x[context]
            cont_emd = torch.sum(cont_emd, dim=1)
        elif self.context_mode == 'add':
            cont_emd = x[context]
            context_weight = self.context_fc(cont_emd)
            cont_emd = torch.sum(context_weight, dim=1)
        elif self.context_mode == 'distance':
            distance = extra_input['distance']
            pos_emd = self.pos_embedding(distance)
            cont_emd = x[context]
            cont_emd += pos_emd
            cont_emd = torch.sum(cont_emd, dim=1)
        emd = torch.cat([sent_emd, cont_emd], dim=1).squeeze()
        emd = self.bn(emd)

        emd = F.dropout(emd, p=0.5, training=self.training)
        h1 = F.relu(self.fc1(emd))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = self.fc2(h1)
        return h2

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


def enhance_dataset(graph, dataset, cut_off=49):
    network = gutil.to_networkx(graph.edge_index, graph.x)
    for sample in dataset.samples:
        sentence = sample['sentence']
        current_path = set()
        for index in range(len(sentence)):
            if sentence[index] != 0:
                current_path.add(sentence[index])
        context = sample['context']
        distance = np.zeros(context.shape[0], dtype=int)
        for index in range(len(context)):
            if context[index] != 0:
                source = context[index]
                min_length = cut_off
                for target in current_path:
                    try:
                        path_length = nx.shortest_path_length(network, source, target)
                    except nx.exception.NetworkXNoPath:
                        path_length = graph.num_nodes
                    if path_length < min_length:
                        min_length = path_length
                distance[index] = min_length
        sample['distance'] = distance
    return dataset


def train(season_id, dm_train_set, dm_test_set, features, edges):

    EMBEDDING_DIM = 200
    batch_size = 128
    epoch_num = 300
    cut_off = 49
    max_acc = 0
    max_v_acc = 0
    model_save_path = './tmp/model_save/gcn_context.model'

    graph = build_graph(features, edges)
    print(graph.num_nodes, graph.num_edges)
    features = torch.FloatTensor(features)
    graph = graph.to(device)

    # dm_valid_set = pickle.load(open(os.path.join('./tmp', season_id, 'unigram_context_valid_dataset.pkl'), 'rb'))
    # dm_valid_set = enhance_dataset(graph, dm_valid_set, cut_off)
    # pickle.dump(dm_valid_set, open(os.path.join('./tmp', season_id, 'unigram_context_valid_dataset.pkl'), 'wb'))

    # dm_train_set = enhance_dataset(graph, dm_train_set, cut_off)
    # dm_test_set = enhance_dataset(graph, dm_test_set, cut_off)
    # pickle.dump(dm_train_set, open(os.path.join('./tmp', season_id, 'unigram_context_train_dataset.pkl'), 'wb'))
    # pickle.dump(dm_test_set, open(os.path.join('./tmp', season_id, 'unigram_context_test_dataset.pkl'), 'wb'))

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

    model = GCNContext(graph, EMBEDDING_DIM, 256, context_mode='distance', cut_off=cut_off)
    model.init_emb(features)
    print(model)
    model.to(device)

    if torch.cuda.is_available():
        print("CUDA : On")
    else:
        print("CUDA : Off")

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=1e-8)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    logging = True
    if logging:
        writer = SummaryWriter()
        log_name = 'gcn_context'

    for epoch in tqdm(range(epoch_num)):
        model.train(mode=True)
        scheduler.step()
        for batch_idx, sample_dict in enumerate(dm_dataloader):
            sentence = torch.LongTensor(sample_dict['sentence'])
            context = torch.LongTensor(sample_dict['context'])
            label = torch.LongTensor(sample_dict['label'])
            distance = torch.LongTensor(sample_dict['distance'])

            sentence = sentence.to(device)
            context = context.to(device)
            label = label.to(device)
            distance = distance.to(device)

            optimizer.zero_grad()
            pred = model.forward(sentence, context, distance=distance)
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
        accuracy = valid_util.validate(model, dm_test_set, dm_test_dataloader, mode='output',
                                       type='graph_context')
        if accuracy > max_acc:
            max_acc = accuracy

        if logging:
            result_dict = valid_util.validate(model, dm_test_set, dm_test_dataloader, mode='report',
                                              type='graph_context')
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

        # v_acc = valid_util.validate(model, dm_valid_set, mode='output', type='graph_context')
        # if v_acc > max_v_acc:
        #     max_v_acc = v_acc

    if logging:
        writer.close()
    print("Max Accuracy: %4.6f" % max_acc)
    # print("Max Validation Accuracy: %4.6f" % max_v_acc)
    return

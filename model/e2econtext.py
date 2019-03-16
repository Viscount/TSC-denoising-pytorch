#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import pickle
import numpy as np
import torch.utils.data as data
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import util.validation as valid_util
from tensorboardX import SummaryWriter

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


class EmbeddingContextModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_words):
        super(EmbeddingContextModeler, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # self.context_weight = Parameter(torch.Tensor(context_words))
        self.context_fc = WeightLayer(context_words, bias=False)
        self.fc1 = nn.Linear(embedding_dim * 2, 256, bias=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.fc3 = nn.Linear(128, 2, bias=True)
        self.embedding_dim = embedding_dim

    def init_emb(self, pre_train_weight):
        init_range = 1 / self.embedding_dim
        if pre_train_weight.shape == self.embedding.weight.data.shape:
            pre_train_weight[1, :] = np.random.uniform(-init_range, init_range, pre_train_weight.shape[1])
            pre_train_weight = torch.FloatTensor(pre_train_weight)
            self.embedding.weight.data = pre_train_weight
            # self.embedding = nn.Embedding.from_pretrained(pre_train_weight, freeze=True)
        else:
            print('Weight data shape mismatch, using default init.')
        # self.context_weight.data = torch.FloatTensor(np.random.uniform(0, init_range, 50))
        return

    def forward(self, sentence, context):
        sent_emd = self.embedding(sentence)
        sent_emd = torch.sum(sent_emd, dim=1)
        cont_emd = self.embedding(context)
        # context_count = (context_count / context_count.sum()).unsqueeze(2)
        # context_weight = self.context_weight.unsqueeze(0)
        # context_weight = context_weight.unsqueeze(2)
        context_weight = self.context_fc(cont_emd)
        context_embed = torch.sum(context_weight, dim=1)
        emd = torch.cat([sent_emd, context_embed], dim=1).squeeze()

        emd = F.dropout(emd, p=0.5, training=self.training)

        h1 = F.relu(self.fc1(emd))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(self.fc2(h1))
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h3 = self.fc3(h2)
        return h3


def train(season_id, dm_train_set, dm_test_set):

    EMBEDDING_DIM = 200
    batch_size = 128
    epoch_num = 100
    max_acc = 0
    max_v_acc = 0
    model_save_path = '.tmp/model_save/straight_embed_context.model'

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

    model = EmbeddingContextModeler(dm_train_set.vocab_size(), EMBEDDING_DIM, dm_train_set.context_words)
    print(model)
    # init_weight = np.loadtxt(os.path.join('./tmp', season_id, 'unigram_weights.txt'))
    # model.init_emb(init_weight)
    model.to(device)

    if torch.cuda.is_available():
        print("CUDA : On")
    else:
        print("CUDA : Off")
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

    logging = False
    if logging:
        log_name = 'straight_embed_context'
        writer = SummaryWriter()

    for epoch in range(epoch_num):
        model.train(mode=True)
        # scheduler.step()
        for batch_idx, sample_dict in enumerate(dm_dataloader):
            sentence = torch.LongTensor(sample_dict['sentence'])
            label = torch.LongTensor(sample_dict['label'])
            context = torch.LongTensor(sample_dict['context'])
            context_count = sample_dict['context_count'].float()

            sentence = sentence.to(device)
            label = label.to(device)
            context = context.to(device)
            context_count = context_count.to(device)

            optimizer.zero_grad()
            pred = model.forward(sentence, context)
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
        accuracy = valid_util.validate(model, dm_test_set, dm_test_dataloader, mode='output', type='context')
        if accuracy > max_acc:
            max_acc = accuracy

        if logging:
            result_dict = valid_util.validate(model, dm_test_set, dm_test_dataloader, mode='report', type='context')
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

        # dm_valid_set = pickle.load(open(os.path.join('./tmp', season_id, 'unigram_context_valid_dataset.pkl'), 'rb'))
        # v_acc = valid_util.validate(model, dm_valid_set, mode='output', type='context')
        # if v_acc > max_v_acc:
        #     max_v_acc = v_acc

    if logging:
        writer.close()
    print("Max Accuracy: %4.6f" % max_acc)
    # print("Max Validation Accuracy: %4.6f" % max_v_acc)
    return

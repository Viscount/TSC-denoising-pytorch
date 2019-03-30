#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from torch import nn
import torch.nn.functional as F
import pickle
import numpy as np
import torch.utils.data as data
from torch.autograd import Variable
import torch.optim as optim
import util.validation as valid_util
from tensorboardX import SummaryWriter
from tqdm import tqdm


class EmbeddingE2EModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingE2EModeler, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim, 256, bias=True)
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
        return

    def forward(self, sentence):
        sent_emd = self.embedding(sentence)
        sent_emd = torch.sum(sent_emd, dim=1)
        h1 = F.relu(self.fc1(sent_emd))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(self.fc2(h1))
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h3 = self.fc3(h2)
        return h3


def train(season_id, dm_train_set, dm_test_set):

    EMBEDDING_DIM = 200
    batch_size = 128
    epoch_num = 300
    max_acc = 0
    max_v_acc = 0
    model_save_path = '.tmp/model_save/straight_embed.model'

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

    model = EmbeddingE2EModeler(dm_train_set.vocab_size(), EMBEDDING_DIM)
    print(model)
    # init_weight = np.loadtxt(os.path.join('./tmp', season_id, 'unigram_weights.txt'))
    # model.init_emb(init_weight)
    if torch.cuda.is_available():
        print("CUDA : On")
        model.cuda()
    else:
        print("CUDA : Off")
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    logging = False
    if logging:
        log_name = 'straight_embed'
        writer = SummaryWriter()

    for epoch in tqdm(range(epoch_num)):
        model.train(mode=True)
        for batch_idx, sample_dict in enumerate(dm_dataloader):
            sentence = Variable(torch.LongTensor(sample_dict['sentence']))
            label = Variable(torch.LongTensor(sample_dict['label']))
            if torch.cuda.is_available():
                sentence = sentence.cuda()
                label = label.cuda()

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
        accuracy = valid_util.validate(model, dm_test_set, dm_test_dataloader, mode='output')
        if accuracy > max_acc:
            max_acc = accuracy
            # torch.save(model.state_dict(), model_save_path)
        if logging:
            result_dict = valid_util.validate(model, dm_test_set, dm_test_dataloader, mode='report')
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

        # dm_valid_set = pickle.load(open(os.path.join('./tmp', season_id, 'unigram_valid_dataset.pkl'), 'rb'))
        # v_acc = valid_util.validate(model, dm_valid_set, mode='output')
        # if v_acc > max_v_acc:
        #     max_v_acc = v_acc

    if logging:
        writer.close()
    print("Max Accuracy: %4.6f" % max_acc)
    # print("Max Validation Accuracy: %4.6f" % max_v_acc)
    return

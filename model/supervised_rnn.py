#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pickle
import torch.utils.data as data
from torch.autograd import Variable
import torch.optim as optim
import util.validation as valid_util
from tensorboardX import SummaryWriter

from model.e2ernn import E2ERNNModeler


def train(dm_train_set, dm_test_set):
    torch.manual_seed(1)

    EMBEDDING_DIM = 200
    hidden_size = 100
    batch_size = 128
    epoch_num = 100
    RNN_type = 'GRU'

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

    model = E2ERNNModeler(dm_train_set.vocab_size(), EMBEDDING_DIM, hidden_size, RNN_type)
    print(model)
    # init_weight = np.loadtxt("./tmp/unigram_weights.txt")
    # model.init_emb(init_weight)
    if torch.cuda.is_available():
        print("CUDA : On")
        model.cuda()
    else:
        print("CUDA : Off")

    embedding_params = list(map(id, model.embedding.parameters()))
    other_params = filter(lambda p: id(p) not in embedding_params, model.parameters())

    optimizer = optim.Adam([
                {'params': other_params},
                {'params': model.embedding.parameters(), 'lr': 1e-3}
            ], lr=1e-3, betas=(0.9, 0.99))

    logging = True
    if logging:
        writer = SummaryWriter()
        log_name = 'Direct_'+RNN_type

    history = None

    for epoch in range(epoch_num):
        for batch_idx, sample_dict in enumerate(dm_dataloader):
            sentence = Variable(torch.LongTensor(sample_dict['sentence']))
            label = Variable(torch.LongTensor(sample_dict['label']))
            if torch.cuda.is_available():
                sentence = sentence.cuda()
                label = label.cuda()

            optimizer.zero_grad()
            pred = model.forward(sentence)
            cross_entropy = nn.NLLLoss()
            loss = cross_entropy(F.log_softmax(pred, dim=1), label)
            if batch_idx % 10 == 0:
                accuracy = valid_util.running_accuracy(pred, label)
                print('epoch: %d batch %d : loss: %4.6f accuracy: %4.6f' % (epoch, batch_idx, loss.item(), accuracy))
                if logging:
                    writer.add_scalar(log_name + '_data/loss', loss.item(), epoch * 10 + batch_idx // 10)
            loss.backward()
            optimizer.step()

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
            writer.add_scalar(log_name + '_data/accuracy', result_dict['accuracy'], epoch)
        valid_util.validate(model, dm_test_set, dm_test_dataloader, mode='output')

        dm_valid_set = pickle.load(open('./tmp/unigram_valid_dataset.pkl', 'rb'))
        valid_util.validate(model, dm_valid_set, mode='output')
    if logging:
        writer.close()
    return

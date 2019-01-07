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
import util.strategy as stg
from tensorboardX import SummaryWriter


class E2ECNNModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, feature_dim, window_sizes, max_len):
        super(E2ECNNModeler, self).__init__()
        self.embedding_dim = embedding_dim
        self.static_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dynamic_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.convs = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=embedding_dim,
                                        out_channels=feature_dim,
                                        kernel_size=h),
                              nn.BatchNorm1d(feature_dim, momentum=0.5),
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=max_len-h+1))
                for h in window_sizes])

        self.fc1 = nn.Linear(feature_dim * len(window_sizes) * 2, 256, bias=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.fc3 = nn.Linear(128, 2, bias=True)

    def init_emb(self, pre_train_weight):
        init_range = 1 / self.embedding_dim
        if pre_train_weight.shape == self.dynamic_embedding.weight.data.shape:
            pre_train_weight[1:] = np.random.uniform(-init_range, init_range, pre_train_weight.shape[1])
            pre_train_weight = torch.FloatTensor(pre_train_weight)
            self.static_embedding = nn.Embedding.from_pretrained(pre_train_weight, freeze=True)
            self.dynamic_embedding.weight.data = pre_train_weight
        return

    def embed(self, sentence):
        sent_emd = self.dynamic_embedding(sentence)
        sent_emd = sent_emd.permute(0, 2, 1)
        out = [conv(sent_emd) for conv in self.convs]
        out = torch.cat(out, dim=1).squeeze()

        sent_static_emd = self.static_embedding(sentence)
        sent_static_emd = sent_static_emd.permute(0, 2, 1)
        out_ = [conv(sent_static_emd) for conv in self.convs]
        out_ = torch.cat(out_, dim=1).squeeze()

        out = torch.cat([out, out_], dim=1)
        return out

    def forward(self, sentence):
        sent_emd = self.embed(sentence)
        h1 = F.relu(self.fc1(sent_emd))
        h1 = F.dropout(h1, p=0.5)
        h2 = F.relu(self.fc2(h1))
        h2 = F.dropout(h2, p=0.5)
        h3 = self.fc3(h2)
        return h3


def train(dm_train_set, dm_test_set):
    torch.manual_seed(1)

    EMBEDDING_DIM = 200
    feature_dim = 50
    max_len = 49
    windows_size = [1, 2, 3, 4]
    batch_size = 128
    epoch_num = 20

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

    model = E2ECNNModeler(dm_train_set.vocab_size(), EMBEDDING_DIM, feature_dim, windows_size, max_len)
    print(model)
    init_weight = np.loadtxt("./tmp/we_weights.txt")
    model.init_emb(init_weight)
    if torch.cuda.is_available():
        print("CUDA : On")
        model.cuda()
    else:
        print("CUDA : Off")

    embedding_params = list(map(id, model.dynamic_embedding.parameters()))
    other_params = filter(lambda p: id(p) not in embedding_params, model.parameters())

    optimizer = optim.Adam([
                {'params': other_params},
                {'params': model.dynamic_embedding.parameters(), 'lr': 1e-4}
            ], lr=1e-3, betas=(0.9, 0.99))

    logging = True
    if logging:
        writer = SummaryWriter()

    history = None

    for epoch in range(epoch_num):

        if (epoch+1) % 3 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5

        for batch_idx, sample_dict in enumerate(dm_dataloader):
            anchor = Variable(torch.LongTensor(sample_dict['anchor']))
            pos = Variable(torch.LongTensor(sample_dict['pos']))
            neg = Variable(torch.LongTensor(sample_dict['neg']))
            label = Variable(torch.LongTensor(sample_dict['label']))
            mask = Variable(torch.LongTensor(sample_dict['mask']))
            mask_ = mask.type(torch.FloatTensor).view(-1)
            if torch.cuda.is_available():
                anchor = anchor.cuda()
                pos = pos.cuda()
                neg = neg.cuda()
                label = label.cuda()
                mask = mask.cuda()
                mask_ = mask_.cuda()

            optimizer.zero_grad()
            anchor_embed = model.embed(anchor)
            pos_embed = model.embed(pos)
            neg_embed = model.embed(neg)
            triplet_loss = nn.TripletMarginLoss(margin=10, p=2)
            embedding_loss = triplet_loss(anchor_embed, pos_embed, neg_embed)
            anchor_pred = model.forward(anchor).unsqueeze(1)
            pos_pred = model.forward(pos).unsqueeze(1)
            neg_pred = model.forward(neg).unsqueeze(1)
            final_pred = torch.cat((anchor_pred, pos_pred, neg_pred), dim=1)
            final_pred = final_pred.view(1, -1, 2)
            final_pred = final_pred.squeeze()

            cross_entropy = nn.NLLLoss(reduction='none')
            label = label.mul(mask)
            label = label.view(-1)
            classify_loss = cross_entropy(F.log_softmax(final_pred, dim=1), label)
            classify_loss = classify_loss.mul(mask_)
            if mask_.sum() > 0:
                classify_loss = classify_loss.sum() / mask_.sum()
            else:
                classify_loss = classify_loss.sum()

            alpha = stg.dynamic_alpha(embedding_loss, classify_loss)
            loss = alpha * embedding_loss + (1-alpha) * classify_loss
            # loss = classify_loss
            # loss = embedding_loss

            if batch_idx % 1000 == 0:
                accuracy = valid_util.running_accuracy(final_pred, label, mask_)
                print('epoch: %d batch %d : loss: %4.6f embed-loss: %4.6f class-loss: %4.6f accuracy: %4.6f'
                      % (epoch, batch_idx, loss.item(), embedding_loss.item(), classify_loss.item(), accuracy))
                if logging:
                    writer.add_scalars('cnn_data/loss', {
                        'Total Loss': loss,
                        'Embedding Loss': embedding_loss,
                        'Classify Loss': classify_loss
                    }, epoch * 10 + batch_idx // 1000)
            loss.backward()
            optimizer.step()

        if logging:
            result_dict = valid_util.validate(model, dm_test_set, dm_test_dataloader, mode='report')
            writer.add_scalars('cnn_data/0-PRF', {
                '0-Precision': result_dict['0']['precision'],
                '0-Recall': result_dict['0']['recall'],
                '0-F1-score': result_dict['0']['f1-score']
            }, epoch)
            writer.add_scalars('cnn_data/1-PRF', {
                '1-Precision': result_dict['1']['precision'],
                '1-Recall': result_dict['1']['recall'],
                '1-F1-score': result_dict['1']['f1-score']
            }, epoch)
            writer.add_scalar('cnn_data/accuracy', result_dict['accuracy'], epoch)
        history = valid_util.validate(model, dm_test_set, dm_test_dataloader, mode='detail', pred_history=history)
        pickle.dump(history, open('./tmp/e2e_cnn_history.pkl', 'wb'))

        # dm_valid_set = pickle.load(open('./tmp/e2e_we_valid_dataset.pkl', 'rb'))
        # valid_util.validate(model, dm_valid_set, mode='output')

    if logging:
        writer.close()
    return

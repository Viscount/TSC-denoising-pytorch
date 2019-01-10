#!/usr/bin/env python
# -*- coding: utf-8 -*-

from util.word_segment import word_segment
import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
import torch.optim as optim
import pickle
import torch.utils.data as data
import numpy as np


class TSCEmbedLanguageModeler(nn.Module):

    def __init__(self, tsc_size, embedding_dim):
        super(TSCEmbedLanguageModeler, self).__init__()
        self.u_embeddings = nn.Embedding(tsc_size, 2000, sparse=True)
        self.fc1 = nn.Linear(2000, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, embedding_dim)
        self.embedding_dim = embedding_dim
        self.init_emb()

    def init_emb(self):
        initrange = 100 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        # self.v_embeddings.weight.data.uniform_(-initrange, initrange)

    def embed(self, u):
        raw_embed = self.u_embeddings(u)
        out1 = f.tanh(self.fc1(raw_embed))
        out1 = f.dropout(out1, p=0.5)
        out2 = f.tanh(self.fc2(out1))
        out2 = f.dropout(out2, p=0.5)
        out = f.tanh(self.out(out2))
        return out

    def forward(self, u_pos, v_pos, v_neg, batch_size):
        embed_u = self.embed(u_pos)
        embed_v = self.embed(v_pos)

        sim_pos = f.cosine_similarity(embed_u, embed_v, dim=1)
        score_pos = torch.exp(sim_pos)

        neg_embed_v = self.embed(v_neg)
        neg_pos = f.cosine_similarity(embed_u, neg_embed_v, dim=1)
        score_neg = torch.exp(neg_pos)

        loss = -(score_pos / (score_pos + score_neg))

        # triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        # loss = triplet_loss(embed_u, embed_v, neg_embed_v)

        return loss.sum() / batch_size

    def save_emb(self, path, rawid_to_ix):
        embed_dict = dict()
        with open(path, 'wb') as f:
            for raw_id in rawid_to_ix:
                ix = Variable(torch.LongTensor([rawid_to_ix[raw_id]]))
                if torch.cuda.is_available():
                    ix = ix.cuda()
                embed = self.embed(ix).cpu().detach().numpy()
                embed_dict[raw_id] = np.squeeze(embed)
            pickle.dump(embed_dict, f)
        return


def common_words(content_a, content_b):
    word_set = set(content_a)
    count = 0
    for word in content_b:
        if word in word_set:
            count += 1
    return count


def negative_sampling(sample, sorted_dm_samples):
    content = sample['content']
    result = None
    while result is None:
        candidate = np.random.choice(len(sorted_dm_samples), 1)
        sample_ = sorted_dm_samples[candidate[0]]
        if common_words(content, sample_['content']) == 0:
            result = candidate[0]
    return result


class DmDataset(data.Dataset):
    def __init__(self, sorted_aggr_samples, context_size, common_words_min_count):
        self.rawId_to_ix = dict()
        self.dm_samples = []

        print('building samples...')
        self.samples = []
        for episode_lvl_samples in sorted_aggr_samples:
            context_start_index = 0
            ix = 0
            for sample in episode_lvl_samples:
                self.dm_samples.append(sample)
                self.rawId_to_ix[sample['raw_id']] = ix
                playback_time = sample['playback_time']
                content = sample['content']

                # update context_start_index
                start_sample = episode_lvl_samples[context_start_index]
                while start_sample['playback_time'] < playback_time - context_size:
                    context_start_index += 1
                    start_sample = episode_lvl_samples[context_start_index]
                # build sample pair in context
                it = context_start_index
                sample_ = episode_lvl_samples[it]
                while sample_['playback_time'] <= playback_time + context_size:
                    if sample['raw_id'] != sample_['raw_id'] and \
                            common_words(content, sample_['content']) >= common_words_min_count:
                        self.samples.append((sample['raw_id'], sample_['raw_id']))
                    it += 1
                    if it >= len(episode_lvl_samples):
                        break
                    else:
                        sample_ = episode_lvl_samples[it]
                ix += 1

        self.dm_size = len(self.dm_samples)
        print('%d samples constructed.' % len(self.samples))
        return

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.rawId_to_ix[sample[0]]
        context = self.rawId_to_ix[sample[1]]
        neg_sample = negative_sampling(self.dm_samples[target], self.dm_samples)
        sample_dict = {
            'pos_u': target,
            'pos_v': context,
            'neg_v': neg_sample
        }
        return sample_dict

    def __len__(self):
        return len(self.samples)


def build_dataset(danmaku_complete):
    danmaku_selected = danmaku_complete[danmaku_complete['season_id'] == '24581']
    grouped = danmaku_selected.groupby('episode_id')

    context_size = 2.5
    common_words_min_count = 2
    neg_sampling_num = 10
    samples = []

    for episode_id, group_data in grouped:
        group_data = group_data.sort_values(by='playback_time')
        episode_lvl_samples = []
        print('Processing Episode %d' % episode_id)
        for index, row in group_data.iterrows():
            content = row['content']
            words = word_segment(str(content))
            sample = {
                'raw_id': row['tsc_raw_id'],
                'content': words,
                'playback_time': row['playback_time']
            }
            episode_lvl_samples.append(sample)
        samples.append(episode_lvl_samples)

    dm_set = DmDataset(samples, context_size, common_words_min_count)
    return dm_set


def train(dm_set):

    EMBEDDING_DIM = 200
    batch_size = 128
    epoch_num = 10

    dm_dataloader = data.DataLoader(
        dataset=dm_set,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=8
    )

    model = TSCEmbedLanguageModeler(dm_set.dm_size, EMBEDDING_DIM)
    print(model)
    if torch.cuda.is_available():
        print("CUDA : On")
        model = model.cuda()
    else:
        print("CUDA : Off")
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    for epoch in range(epoch_num):
        for batch_idx, sample in enumerate(dm_dataloader):
            pos_u = Variable(torch.LongTensor(sample['pos_u']))
            pos_v = Variable(torch.LongTensor(sample['pos_v']))
            neg_v = Variable(torch.LongTensor(sample['neg_v']))

            if torch.cuda.is_available():
                pos_u = pos_u.cuda()
                pos_v = pos_v.cuda()
                neg_v = neg_v.cuda()

            optimizer.zero_grad()
            loss = model(pos_u, pos_v, neg_v, batch_size)
            if batch_idx % 100 == 0:
                print('epoch: %d batch %d : loss: %4.6f' % (epoch, batch_idx, loss.item()))

            loss.backward()

            optimizer.step()

    model.save_emb('tmp/dm_embeds.p', dm_set.rawId_to_ix)

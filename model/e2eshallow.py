#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
import pickle
import torch.utils.data as data
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import pandas as pd
import collections
from util.word_segment import word_segment
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class EmbeddingE2EModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingE2EModeler, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 256, bias=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.fc3 = nn.Linear(128, 2, bias=True)

    def init_emb(self):
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, sentence):
        sent_emd = self.embedding(sentence)
        size = sent_emd.shape[1]
        sent_emd = torch.sum(sent_emd, dim=1) / size
        h1 = F.relu(self.fc1(sent_emd))
        h1 = F.dropout(h1, p=0.5)
        h2 = F.relu(self.fc2(h1))
        h2 = F.dropout(h2, p=0.5)
        h3 = F.sigmoid(self.fc3(h2))
        out = F.softmax(h3, dim=1)
        return out

    def save_emb(self, path, word_to_ix):
        embeds = self.u_embeddings.weight.data
        embed_dict = dict()
        with open(path, 'wb') as f:
            for word in word_to_ix:
                ix = word_to_ix[word]
                embed_dict[word] = embeds[ix].cpu().numpy()
            pickle.dump(embed_dict, f)
        return


class DmDataset(data.Dataset):
    def __init__(self, dm_samples, min_count, max_len, dictionary=None):
        self.max_len = max_len
        if dictionary is not None:
            self.word_to_ix = dictionary
        else:
            print('building vocabulary...')
            aggregate_sample = []
            for sample in dm_samples:
                aggregate_sample.extend(sample['words'])
            counter = {'UNK': 0}
            counter.update(collections.Counter(aggregate_sample).most_common())
            rare_words = set()
            for word in counter:
                if word != 'UNK' and counter[word] <= min_count:
                    rare_words.add(word)
            for word in rare_words:
                counter['UNK'] += counter[word]
                counter.pop(word)
            print('%d words founded in vocabulary' % len(counter))

            self.vocab_counter = counter
            self.word_to_ix = {
                'EPT': 0
            }
            for word in counter:
                self.word_to_ix[word] = len(self.word_to_ix)

        print('building samples...')
        self.samples = []
        self.labels = []
        for sample in dm_samples:
            label = sample['label']
            sample_ = np.zeros(max_len, dtype=int)
            index = 0
            for word in sample['words']:
                sample_[index] = self.word2ix(word)
                index += 1
            self.samples.append(np.array(sample_))
            self.labels.append(label)

        print('%d samples constructed.' % len(self.samples))
        return

    def word2ix(self, word):
        if word in self.word_to_ix:
            return self.word_to_ix[word]
        else:
            return self.word_to_ix['UNK']

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return len(self.samples)

    def save_vocab(self, path):
        vocab = []
        for word in self.vocab_counter:
            vocab.append({'idx': self.word_to_ix[word],
                          'word': word,
                          'count': self.vocab_counter[word]})
        df = pd.DataFrame(vocab)
        df.to_csv(path, index=False)
        return

    def vocab_size(self):
        return len(self.word_to_ix)


def build_dataset(danmaku_complete):
    min_count = 2
    danmaku_selected = danmaku_complete[danmaku_complete['season_id'] == '24581']
    samples = []
    count = 0
    max_len = 0

    for index, row in danmaku_selected.iterrows():
        count += 1
        content = row['content']
        words = word_segment(str(content))
        if len(words) > max_len:
            max_len = len(words)
        if row['block_level'] >= 9:
            label = 0
            samples.append({
                'raw_id': row['tsc_raw_id'],
                'words': words,
                'label': label
            })
        elif 0 < row['block_level'] <= 2:
            label = 1
            samples.append({
                'raw_id': row['tsc_raw_id'],
                'words': words,
                'label': label
            })
        else:
            continue
        if count % 10000 == 0:
            print("%d danmakus processed." % count)

    train_samples, test_samples = train_test_split(samples, test_size=0.25, shuffle=True)
    dm_train_set = DmDataset(train_samples, min_count, max_len)
    dm_test_set = DmDataset(test_samples, min_count, max_len, dictionary=dm_train_set.word_to_ix)
    return dm_train_set, dm_test_set


def train(dm_train_set, dm_test_set):
    torch.manual_seed(1)

    EMBEDDING_DIM = 200
    batch_size = 128
    epoch_num = 6

    dm_dataloader = data.DataLoader(
        dataset=dm_train_set,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=8
    )

    dm_test_dataloader = data.DataLoader(
        dataset=dm_test_set,
        batch_size=128,
        shuffle=True,
        drop_last=False,
        num_workers=8
    )

    model = EmbeddingE2EModeler(dm_train_set.vocab_size(), EMBEDDING_DIM)
    print(model)
    if torch.cuda.is_available():
        print("CUDA : On")
        model.cuda()
    else:
        print("CUDA : Off")
    optimizer = optim.Adam(model.parameters(), lr=1e-6, betas=(0.9, 0.99))

    for epoch in range(epoch_num):
        for batch_idx, (sentence, label) in enumerate(dm_dataloader):
            sentence = Variable(torch.LongTensor(sentence))
            label = Variable(torch.LongTensor(label))
            if torch.cuda.is_available():
                sentence = sentence.cuda()
                label = label.cuda()

            optimizer.zero_grad()
            pred = model.forward(sentence)
            cross_entropy = nn.CrossEntropyLoss()
            loss = cross_entropy(pred, label)
            if batch_idx % 10 == 0:
                print('epoch: %d batch %d : loss: %4.6f' % (epoch, batch_idx, loss.item()))
            loss.backward()
            optimizer.step()
        pred_array = []
        label_array = []
        for batch_idx, (sentence, label) in enumerate(dm_test_dataloader):
            sentence = Variable(torch.LongTensor(sentence))
            if torch.cuda.is_available():
                sentence = sentence.cuda()
            pred = model.forward(sentence)
            pred_array.extend(pred.argmax(dim=1).cpu().numpy())
            label_array.extend(label.numpy())
        print(classification_report(label_array, pred_array))

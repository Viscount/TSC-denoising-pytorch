import torch
from torch import nn
import torch.nn.functional as F
import pickle
import torch.utils.data as data
from torch.autograd import Variable
import torch.optim as optim
import math
import numpy as np
import pandas as pd
import random
import collections
from util.word_segment import word_segment
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class EmbeddingE2EModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingE2EModeler, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.fc1 = nn.Linear(embedding_dim, 256, bias=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.fc3 = nn.Linear(128, 64, bias=True)

    def init_emb(self):
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, sentence):
        sent_emd = self.embedding(sentence)
        sent_emd = torch.sum(sent_emd, dim=1)
        h1 = F.relu(self.fc1(sent_emd))
        h1 = F.dropout(h1, p=0.5)
        h2 = F.relu(self.fc2(h1))
        h2 = F.dropout(h2, p=0.5)
        h3 = F.relu(self.fc3(h2))
        h3 = F.dropout(h3, p=0.5)
        out = F.softmax(h3, 2)
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
    def __init__(self, dm_samples, min_count):
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
        self.word_to_ix = dict()
        for word in counter:
            self.word_to_ix[word] = len(self.word_to_ix)

        print('building samples...')
        self.samples = []
        self.labels = []
        for sample in dm_samples:
            label = np.array([0., 0.])
            label[sample['label']] = 1.
            sample_ = [self.word2ix(word) for word in sample['words']]
            self.samples.append(sample_)
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
        return len(self.vocab_counter)


def build_dataset(danmaku_complete):
    min_count = 5
    danmaku_selected = danmaku_complete[danmaku_complete['season_id'] == '24581']
    samples = []
    count = 0

    for index, row in danmaku_selected.iterrows():
        count += 1
        content = row['content']
        words = word_segment(str(content))
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
        if count % 100000 == 0:
            print("%d danmakus processed." % count)

    train_samples, test_samples = train_test_split(samples, test_size=0.25, shuffle=True)
    dm_train_set = DmDataset(train_samples, min_count)
    dm_test_set = DmDataset(train_samples, min_count)
    return dm_train_set, dm_test_set


def train(dm_train_set, dm_test_set):
    torch.manual_seed(1)

    EMBEDDING_DIM = 200
    batch_size = 128
    epoch_num = 20

    dm_dataloader = data.DataLoader(
        dataset=dm_train_set,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=8
    )

    dm_test_dataloader = data.DataLoader(
        dataset=dm_test_set,
        shuffle=True,
        drop_last=True,
        num_workers=8
    )

    model = EmbeddingE2EModeler(dm_train_set.vocab_size(), EMBEDDING_DIM)
    print(model)
    if torch.cuda.is_available():
        print("CUDA : On")
        model.cuda()
    else:
        print("CUDA : Off")
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epoch_num):
        for batch_idx, (sentence, label) in enumerate(dm_dataloader):
            sentence_ = Variable(torch.LongTensor(sentence))
            label_ = Variable(torch.LongTensor(label))
            if torch.cuda.is_available():
                sentence_ = sentence_.cuda()
                label_ = label_.cuda()

            optimizer.zero_grad()
            pred = model.forward(sentence_)
            loss = F.cross_entropy(pred, label_)
            if batch_idx % 100 == 0:
                print('epoch: %d batch %d : loss: %4.6f' % (epoch, batch_idx, loss.item()))
            loss.backward()
            optimizer.step()
        pred_array = []
        label_array = []
        for (sentence, label) in enumerate(dm_test_dataloader):
            sentence_ = Variable(torch.LongTensor(sentence))
            if torch.cuda.is_available():
                sentence_ = sentence_.cuda()
            pred = model.forward(sentence_)
            pred_array.append(pred.argmax(dim=1))
            label_array.append(label)
        print(classification_report(label_array, pred_array))

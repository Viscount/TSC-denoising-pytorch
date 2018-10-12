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


class SkipGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramLanguageModeler, self).__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.embedding_dim = embedding_dim
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, u_pos, v_pos, v_neg, batch_size):
        embed_u = self.u_embeddings(u_pos)
        embed_v = self.v_embeddings(v_pos)

        score = torch.mul(embed_u, embed_v)
        score = torch.sum(score, dim=1)
        log_target = F.logsigmoid(score).squeeze()

        neg_embed_v = self.v_embeddings(v_neg)

        neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        sum_log_sampled = F.logsigmoid(-1 * neg_score).squeeze()

        loss = log_target + sum_log_sampled

        return -1 * loss.sum() / batch_size

    def save_emb(self, path, word_to_ix):
        embeds = self.u_embeddings.weight.data
        embed_dict = dict()
        with open(path, 'wb') as f:
            for word in word_to_ix:
                ix = word_to_ix[word]
                embed_dict[word] = embeds[ix].numpy()
            pickle.dump(embed_dict, f)
        return


class DmDataset(data.Dataset):
    def __init__(self, dm_samples, context_size, min_count, neg_sampling_num):
        self.neg_sampling_num = neg_sampling_num
        print('building vocabulary...')
        aggregate_sample = []
        for sample in dm_samples:
            aggregate_sample.extend(sample)
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

        counts = [self.vocab_counter[key] for key in self.vocab_counter]
        frequency = np.array(counts) / sum(counts)
        self.subsampling_P = dict()
        for idx, x in enumerate(frequency):
            y = (math.sqrt(x / 0.001) + 1) * 0.001 / x
            self.subsampling_P[idx] = y

        pow_frequency = np.array(counts) ** 0.75
        power = sum(pow_frequency)
        self.neg_sampling_ratio = pow_frequency / power

        print('building samples...')
        self.samples = []
        span = context_size * 2 + 1
        for sample in dm_samples:
            # skip heading and tailing words
            # subsampling
            sample_ = []
            for word in sample:
                word_ix = self.word2ix(word)
                if random.random() < self.subsampling_P[word_ix]:
                    sample_.append(word_ix)
            # generate word pair
            start_index = 0
            done = False
            while start_index + span <= len(sample_):
                buffer = sample_[start_index: start_index + span]
                done = True
                target_word = buffer[context_size]
                for index in range(0, len(buffer)):
                    if index != context_size:
                        self.samples.append((target_word, buffer[index]))
                start_index += 1
            if not done:
                buffer = sample_[:]
                if len(buffer) > 1:
                    target_word = buffer[len(buffer) // 2]
                    for index in range(0, len(buffer)):
                        if index != len(buffer) // 2:
                            self.samples.append((target_word, buffer[index]))

        print('%d samples constructed.' % len(self.samples))
        return

    def word2ix(self, word):
        if word in self.word_to_ix:
            return self.word_to_ix[word]
        else:
            return self.word_to_ix['UNK']

    def __getitem__(self, index):
        sample = self.samples[index]
        target = sample[0]
        context = sample[1]
        neg_samples = np.random.choice(len(self.vocab_counter), self.neg_sampling_num,
                                       p=self.neg_sampling_ratio)
        sample_dict = {
            'pos_u': target,
            'pos_v': context,
            'neg_v': np.array(neg_samples)
        }
        return sample_dict

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
    context_size = 3
    min_count = 5
    neg_sampling_num = 10
    samples = []
    count = 0

    for index, row in danmaku_complete.iterrows():
        count += 1
        content = row['content']
        words = word_segment(str(content))
        samples.append(words)
        if count % 100000 == 0:
            print("%d danmakus processed." % count)

    dm_set = DmDataset(samples, context_size, min_count, neg_sampling_num)
    return dm_set


def train(dm_set):
    torch.manual_seed(1)

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

    model = SkipGramLanguageModeler(dm_set.vocab_size(), EMBEDDING_DIM)
    print(model)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.3)

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
            print('epoch: %d batch %d : loss: %4.4f' % (epoch, batch_idx, loss.data[0]))

            loss.backward()

            optimizer.step()

    dm_set.save_vocab('tmp/skip_gram_vocab.csv')
    model.save_emb('tmp/skip_gram_embeds.p', dm_set.word_to_ix)
    return

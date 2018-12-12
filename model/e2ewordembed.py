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
import util.validation as valid_util
import util.strategy as stg
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter


class EmbeddingE2EModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingE2EModeler, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim, 256, bias=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.fc3 = nn.Linear(128, 2, bias=True)

    def init_emb(self, pre_train_weight):
        init_range = 1 / self.embedding_dim
        if pre_train_weight.shape == self.embedding.weight.data.shape:
            pre_train_weight[1:] = np.random.uniform(-init_range, init_range, pre_train_weight.shape[1])
            pre_train_weight = torch.FloatTensor(pre_train_weight)
            self.embedding.weight.data = pre_train_weight
        return

    def embed(self, sentence):
        sent_emd = self.embedding(sentence)
        sent_emd = torch.sum(sent_emd, dim=1)
        return sent_emd

    def forward(self, sentence):
        sent_emd = self.embed(sentence)
        h1 = F.relu(self.fc1(sent_emd))
        h1 = F.dropout(h1, p=0.5)
        h2 = F.relu(self.fc2(h1))
        h2 = F.dropout(h2, p=0.5)
        h3 = self.fc3(h2)
        return h3


class DmTrainDataset(data.Dataset):
    def __init__(self, dm_samples, min_count, max_len, context_size, min_common_words, dictionary=None):
        self.max_len = max_len
        self.min_count = min_count
        self.min_common_words = min_common_words
        all_sentences = []
        if dictionary is not None:
            self.word_to_ix = dictionary
        else:
            print('building vocabulary...')
            aggregate_samples = []
            for episode_lvl_samples in dm_samples:
                for sample in episode_lvl_samples:
                    all_sentences.append(sample)
                    aggregate_samples.extend(sample['content'])
            counter = {'UNK': 0}
            counter.update(collections.Counter(aggregate_samples).most_common())
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
        for episode_lvl_samples in dm_samples:
            context_start_index = 0
            for sample in episode_lvl_samples:

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
                            common_words(content, sample_['content']) >= min_common_words:
                        sentence_anchor = tokenize(sample['content'], max_len, self.word2ix)
                        sentence_positive = tokenize(sample_['content'], max_len, self.word2ix)
                        neg_sample = negative_sampling(sample, all_sentences)
                        sentence_negative = tokenize(neg_sample['content'], max_len, self.word2ix)
                        self.samples.append((sentence_anchor, sentence_positive, sentence_negative))
                        self.labels.append((sample['label'], sample_['label'], neg_sample['label']))
                    it += 1
                    if it >= len(episode_lvl_samples):
                        break
                    else:
                        sample_ = episode_lvl_samples[it]

        print('%d samples constructed.' % len(self.samples))
        return

    def word2ix(self, word):
        if word in self.word_to_ix:
            return self.word_to_ix[word]
        else:
            return self.word_to_ix['UNK']

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        mask = np.zeros(3, dtype=int)
        for i in range(3):
            if label[i] != -1:
                mask[i] = 1
        sample_dict = {
            'anchor': sample[0],
            'pos': sample[1],
            'neg': sample[2],
            'label': np.array([label[0], label[1], label[2]]),
            'mask': mask
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
        return len(self.word_to_ix)


class DmTestDataset(data.Dataset):
    def __init__(self, dm_samples, max_len, dictionary):
        self.max_len = max_len
        self.word_to_ix = dictionary

        print('building samples...')
        self.samples = []
        self.labels = []
        for sample in dm_samples:
            label = sample['label']
            sample_ = tokenize(sample['content'], max_len, self.word2ix)
            self.samples.append((sample['raw_id'], np.array(sample_)))
            self.labels.append(label)
        print('%d samples constructed.' % len(self.samples))
        return

    def word2ix(self, word):
        if word in self.word_to_ix:
            return self.word_to_ix[word]
        else:
            return self.word_to_ix['UNK']

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        sample_dict = {
            'raw_id': sample[0],
            'sentence': sample[1],
            'label': label
        }
        return sample_dict

    def __len__(self):
        return len(self.samples)


def common_words(content_a, content_b):
    word_set = set(content_a)
    count = 0
    for word in content_b:
        if word in word_set:
            count += 1
    return count


def tokenize(sentence, max_len, dictionary_func):
    result = np.zeros(max_len, dtype=int)
    index = 0
    for word in sentence:
        result[index] = dictionary_func(word)
        index += 1
    return result


def negative_sampling(sample, all_samples):
    content = sample['content']
    result = None
    while result is None:
        candidate = np.random.choice(len(all_samples), 1)
        sample_ = all_samples[candidate[0]]
        if common_words(content, sample_['content']) == 0:
            result = sample_
    return result


def build_dataset(danmaku_complete):
    danmaku_selected = danmaku_complete[danmaku_complete['season_id'] == '24581']
    grouped = danmaku_selected.groupby('episode_id')

    context_size = 2.5
    common_words_min_count = 3
    min_count = 3
    max_len = 0
    samples = []
    pos_label_set = set()
    neg_label_set = set()

    # build all danmaku samples
    for episode_id, group_data in grouped:
        group_data = group_data.sort_values(by='playback_time')
        episode_lvl_samples = []
        print('Processing Episode %d' % episode_id)
        for index, row in group_data.iterrows():
            content = row['content']
            words = word_segment(str(content))
            if len(words) > max_len:
                max_len = len(words)
            if row['block_level'] >= 9:
                label = 0
                pos_label_set.add(row['tsc_raw_id'])
            elif 0 < row['block_level'] <= 2:
                label = 1
                neg_label_set.add(row['tsc_raw_id'])
            else:
                label = -1
            sample = {
                'raw_id': row['tsc_raw_id'],
                'content': words,
                'playback_time': row['playback_time'],
                'label': label
            }
            episode_lvl_samples.append(sample)
        samples.append(episode_lvl_samples)

    # train-test split
    pos_train, pos_test = train_test_split(list(pos_label_set), test_size=0.25, shuffle=True)
    neg_tran, neg_test = train_test_split(list(neg_label_set), test_size=0.25, shuffle=True)
    test_select = set()
    test_select.update(pos_test)
    test_select.update(neg_test)

    train_samples = []
    test_samples = []
    for episode_lvl_samples in samples:
        episode_lvl_samples_ = []
        for sample in episode_lvl_samples:
            if sample['raw_id'] in test_select:
                test_samples.append(sample)
            else:
                episode_lvl_samples_.append(sample)
        train_samples.append(episode_lvl_samples_)

    # build Dataset Class
    dm_train_set = DmTrainDataset(train_samples, min_count, max_len, context_size, common_words_min_count)
    dm_test_set = DmTestDataset(test_samples, max_len, dm_train_set.word_to_ix)
    return dm_train_set, dm_test_set


def train(dm_train_set, dm_test_set):
    torch.manual_seed(1)

    EMBEDDING_DIM = 200
    batch_size = 128
    epoch_num = 50

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
        shuffle=True,
        drop_last=False,
        num_workers=8
    )

    model = EmbeddingE2EModeler(dm_train_set.vocab_size(), EMBEDDING_DIM)
    print(model)
    init_weight = np.loadtxt("./tmp/we_weights.txt")
    model.init_emb(init_weight)
    if torch.cuda.is_available():
        print("CUDA : On")
        model.cuda()
    else:
        print("CUDA : Off")

    embedding_params = list(map(id, model.embedding.parameters()))
    other_params = filter(lambda p: id(p) not in embedding_params, model.parameters())

    optimizer = optim.Adam([
                {'params': other_params},
                {'params': model.embedding.parameters(), 'lr': 1e-4}
            ], lr=1e-3, betas=(0.9, 0.99))

    logging = True
    if logging:
        writer = SummaryWriter()

    history = None

    for epoch in range(epoch_num):
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

            cross_entropy = nn.CrossEntropyLoss(reduction='none')
            label = label.mul(mask)
            label = label.view(-1)
            classify_loss = cross_entropy(final_pred, label)
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
                    writer.add_scalars('we_data/loss', {
                        'Total Loss': loss,
                        'Embedding Loss': embedding_loss,
                        'Classify Loss': classify_loss
                    }, epoch * 10 + batch_idx // 1000)
            loss.backward()
            optimizer.step()

        if logging:
            result_dict = valid_util.validate(model, dm_test_set, dm_test_dataloader, mode='report')
            writer.add_scalars('we_data/0-PRF', {
                '0-Precision': result_dict['0']['precision'],
                '0-Recall': result_dict['0']['recall'],
                '0-F1-score': result_dict['0']['f1-score']
            }, epoch)
            writer.add_scalars('we_data/1-PRF', {
                '1-Precision': result_dict['1']['precision'],
                '1-Recall': result_dict['1']['recall'],
                '1-F1-score': result_dict['1']['f1-score']
            }, epoch)
            writer.add_scalar('we_data/accuracy', result_dict['accuracy'], epoch)

        history = valid_util.validate(model, dm_test_set, dm_test_dataloader, mode='detail', pred_history=history)
        pickle.dump(history, open('./tmp/e2e_we_history.pkl', 'wb'))

        # dm_valid_set = pickle.load(open('./tmp/e2e_we_valid_dataset.pkl', 'rb'))
        # valid_util.validate(model, dm_valid_set, mode='output')

    if logging:
        writer.close()
    return

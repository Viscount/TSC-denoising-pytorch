#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
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


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
            q: Queries张量，形状为[B, L_q, D_q]
            k: Keys张量，形状为[B, L_k, D_k]
            v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
            scale: 缩放因子，一个浮点标量
            attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
            上下文张量和attetention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            # 给需要mask的地方设置一个负无穷
            attention = attention.masked_fill_(attn_mask, -np.inf)
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
          query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


class E2ESelfAttentionModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, feature_dim, max_len):
        super(E2ESelfAttentionModeler, self).__init__()
        self.embedding_dim = embedding_dim
        self.static_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dynamic_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.self_attention = MultiHeadAttention(model_dim=embedding_dim)

        self.fc1 = nn.Linear(embedding_dim, 256, bias=True)
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
        sent_emd, attention = self.self_attention.forward(sent_emd, sent_emd, sent_emd, None)
        sent_emd = torch.sum(sent_emd, dim=1)
        return sent_emd

    def forward(self, sentence):
        sent_emd = self.embed(sentence)
        h1 = F.relu(self.fc1(sent_emd))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(self.fc2(h1))
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h3 = self.fc3(h2)
        return h3


def train(season_id, dm_train_set, dm_test_set):

    EMBEDDING_DIM = 200
    feature_dim = 50
    max_len = 49
    batch_size = 256
    epoch_num = 50
    max_acc = 0
    max_v_acc = 0

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

    model = E2ESelfAttentionModeler(dm_train_set.vocab_size(), EMBEDDING_DIM, feature_dim, max_len)
    print(model)
    init_weight = np.loadtxt(os.path.join('./tmp', season_id, 'we_weights.txt'))
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
            ], lr=1e-4, betas=(0.9, 0.99))

    logging = True
    if logging:
        writer = SummaryWriter()
        log_name = 'sa'

    history = None

    for epoch in range(epoch_num):

        model.train(mode=True)
        if epoch > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.8

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

            if batch_idx % 100 == 0:
                accuracy = valid_util.running_accuracy(final_pred, label, mask_)
                print('epoch: %d batch %d : loss: %4.6f embed-loss: %4.6f class-loss: %4.6f accuracy: %4.6f num: %4.1f'
                      % (epoch, batch_idx, loss.item(), embedding_loss.item(), classify_loss.item(), accuracy, mask_.sum()))
                if logging:
                    writer.add_scalars(log_name+'_data/loss', {
                        'Total Loss': loss,
                        'Embedding Loss': embedding_loss,
                        'Classify Loss': classify_loss
                    }, epoch * 10 + batch_idx // 100)
            loss.backward()
            optimizer.step()

        model.eval()
        if logging:
            result_dict = valid_util.validate(model, dm_test_set, dm_test_dataloader, mode='report')
            writer.add_scalars(log_name+'_data/0-PRF', {
                '0-Precision': result_dict['0']['precision'],
                '0-Recall': result_dict['0']['recall'],
                '0-F1-score': result_dict['0']['f1-score']
            }, epoch)
            writer.add_scalars(log_name+'_data/1-PRF', {
                '1-Precision': result_dict['1']['precision'],
                '1-Recall': result_dict['1']['recall'],
                '1-F1-score': result_dict['1']['f1-score']
            }, epoch)
            writer.add_scalar(log_name+'_data/accuracy', result_dict['accuracy'], epoch)
        accuracy, history = valid_util.validate(model, dm_test_set, dm_test_dataloader,
                                                mode='detail', pred_history=history)
        # pickle.dump(history, open('./tmp/e2e_sa_history.pkl', 'wb'))
        if accuracy > max_acc:
            max_acc = accuracy
            # torch.save(model.state_dict(), model_save_path)

        # dm_valid_set = pickle.load(open('./tmp/triplet_valid_dataset.pkl', 'rb'))
        # v_acc = valid_util.validate(model, dm_valid_set, mode='output')
        # if v_acc > max_v_acc:
        #     max_v_acc = v_acc

    if logging:
        writer.close()
    print("Max Accuracy: %4.6f" % max_acc)
    print("Max Validation Accuracy: %4.6f" % max_v_acc)
    return

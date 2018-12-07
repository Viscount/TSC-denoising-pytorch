#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from sklearn.metrics import classification_report


def validate(model, dm_test_set, dataloader=None, mode='acc', py=False):
    if dataloader is None:
        dm_dataloader = data.DataLoader(
            dataset=dm_test_set,
            batch_size=128,
            shuffle=False,
            drop_last=False,
            num_workers=8
        )
    else:
        dm_dataloader = dataloader
    pred_array = []
    label_array = []
    for batch_idx, sample_dict in enumerate(dm_dataloader):
        sentence = Variable(torch.LongTensor(sample_dict['sentence']))
        if torch.cuda.is_available():
            sentence = sentence.cuda()
        if py:
            pinyin = Variable(torch.LongTensor(sample_dict['pinyin']))
            if torch.cuda.is_available():
                pinyin = pinyin.cuda()
            pred = model.forward(sentence, pinyin)
        else:
            pred = model.forward(sentence)

        pred = F.softmax(pred, dim=1)
        pred_array.extend(pred.argmax(dim=1).cpu().numpy())
        label_array.extend(sample_dict['label'].numpy())

        pred_tensor = torch.LongTensor(pred_array)
        label_tensor = torch.LongTensor(label_array)
        count = torch.eq(pred_tensor, label_tensor)
        accuracy = count.sum().item() * 1.0 / count.shape[0]
    if mode == 'acc':
        return accuracy
    elif mode == 'report':
        report_dict = classification_report(label_array, pred_array, output_dict=True)
        report_dict['accuracy'] = accuracy
        return report_dict
    else:
        print('Test Accuracy: %4.6f' % accuracy)
        return classification_report(label_array, pred_array, digits=4)


def running_accuracy(pred, label, mask=None):
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1).squeeze()
    if mask is None:
        hit_count = torch.eq(pred, label)
        return hit_count.sum().item() * 1.0 / hit_count.shape[0]
    else:
        total_count = mask.sum().item()
        hit = 0
        for i in range(mask.shape[0]):
            if mask[i] == 1:
                if pred[i] == label[i]:
                    hit += 1
        return hit * 1.0 / total_count

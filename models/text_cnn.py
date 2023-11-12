# -*- coding:utf-8 -*-
# File       : text_bilstm.py
# Time       ：12/8/2023 下午 11:13
# Author     ：rain
# Description：Text分类 BILSTM网络结构
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes,
                 num_filters=256, filter_sizes=(3, 4, 5),
                 emb_dropout=0.5, pretrained_embedding=None):
        super(TextCNN, self).__init__()
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, num_filters, (k, embedding_size))
                for k in filter_sizes
            ]
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.classifier = nn.Linear(num_filters*len(filter_sizes), num_classes)
        self.criterion = CrossEntropyLoss()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_ids, input_mask, target=None):
        out = self.embedding(input_ids)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        logits = self.classifier(out)
        loss = self.criterion(logits, target)
        return loss, logits

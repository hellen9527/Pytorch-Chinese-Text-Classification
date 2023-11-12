# -*- coding:utf-8 -*-
# File       : text_bilstm.py
# Time       ：12/8/2023 下午 11:13
# Author     ：rain
# Description：Text分类 FastText
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class FastText(nn.Module):
    def __init__(self, vocab_size, gram2_size, gram3_size, embedding_size,
                 hidden_size, num_classes, emb_dropout=0.1, pretrained_embedding=None):
        super(FastText, self).__init__()
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_ngram2 = nn.Embedding(gram2_size, embedding_size)
        self.embedding_ngram3 = nn.Embedding(gram3_size, embedding_size)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.fc1 = nn.Linear(embedding_size * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.criterion = CrossEntropyLoss()

    def forward(self, input_ids, input_mask, gram2_ids, gram3_ids, target=None):
        out_word = self.embedding(input_ids)
        out_bigram = self.embedding_ngram2(gram2_ids)
        out_trigram = self.embedding_ngram3(gram3_ids)
        out = torch.cat((out_word, out_bigram, out_trigram), -1)
        out = out.mean(dim=1)
        out = self.emb_dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        logits = self.fc2(out)
        loss = self.criterion(logits, target)
        return loss, logits

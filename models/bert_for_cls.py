# -*- coding:utf-8 -*-
# File       : text_classify.py
# Time       ：12/8/2023 下午 11:17
# Author     ：rain
# Description：
import torch
import torch.nn as nn
from transformers.models.bert import BertPreTrainedModel, BertModel, BertConfig
from torch.nn import CrossEntropyLoss
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy


class TextBertCLS(BertPreTrainedModel):
    def __init__(self, config: BertConfig, loss_type: str):
        super(TextBertCLS, self).__init__(config)
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = loss_type
        # dropout
        self.lstm_dropout = nn.Dropout(p=0.2)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                targets=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output = outputs[1]
        sequence_output = self.dropout(sequence_output)
        # sequence_output = sequence_output * attention_mask.float().unsqueeze(2)
        # sequence_out_res = torch.mean(sequence_output, dim=1).squeeze()
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if targets is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, targets)
            outputs = (loss,) + outputs
        return outputs

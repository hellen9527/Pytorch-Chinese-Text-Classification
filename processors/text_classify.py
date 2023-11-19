# -*- coding:utf-8 -*-
# File       : text_classify.py
# Time       ：6/8/2023 下午 11:22
# Author     ：rain
# Description：
from abc import ABC

import torch
import logging
import os
import copy
import json

logger = logging.getLogger(__name__)


class DataProcessor(object):
    """Base class for data converters for text classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_vocab(self, vocab_file):
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file, delimiter="\t"):
        """Reads a csv file, tab separated value."""
        return []

    @classmethod
    def _read_text(cls, input_file):
        dlines = []
        with open(input_file, 'r', encoding="utf-8") as fr:
            for line in fr:
                line = line.strip()
                dlines.append(line)
        return dlines

    @classmethod
    def _read_json(cls, input_file):
        return []


class InputExample(object):
    """A single training/test example for text classification."""

    def __init__(self, guid, text_a, label):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            label: str. The classify of text
        """
        self.guid = guid
        self.text_a = text_a
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_len, label_idx,
                 gram2_ids=None, gram3_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_idx = label_idx
        self.input_len = input_len
        if gram2_ids is not None:
            self.gram2_ids = gram2_ids
        if gram3_ids is not None:
            self.gram3_ids = gram3_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    return all_input_ids, all_attention_mask, all_labels, all_lens


def fasttext_collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_lens, all_labels, \
    all_gram2_ids, all_gram3_ids = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_gram2_ids = all_gram2_ids[:, :max_len]
    all_gram3_ids = all_gram3_ids[:, :max_len]
    return all_input_ids, all_attention_mask, all_labels, all_lens, all_gram2_ids, all_gram3_ids


def bert_convert_examples_to_features(examples, label2id, max_seq_length, tokenizer=None,
                                      cls_token="[CLS]", sep_token="[SEP]", pad_token=0,
                                      mask_padding_with_zero=True, ):
    """ Loads a data file into a list of `InputBatch`s
    """
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        if tokenizer is None:
            tokens = example.text_a.split(" ")
        else:
            tokens = tokenizer.tokenize(example.text_a)
        label_id = label2id[example.label]
        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
        tokens = [cls_token] + tokens + [sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = sum(input_mask)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("label_id: %s", example.label)

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask,
                                      input_len=input_len, label_idx=label_id))
    return features


def convert_fasttext_features(examples, max_seq_length, label2id, pad_token=0,
                              vocab_dict=None, gram2_dict=None, gram3_dict=None):
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        word_list = example.text_a.split()
        input_ids = [vocab_dict.get(word, 1) for word in word_list if word.strip()]
        label_id = label2id[example.label]
        input_mask = [1] * len(input_ids)
        gram2_ids, gram3_ids = [], []
        for i, word in enumerate(word_list):
            if i < len(word_list) - 1:
                gram2 = "".join(word_list[i:i+2])
                if gram2 not in gram2_dict:
                    gram2 = "[UNK]"
                gram2_id = gram2_dict[gram2]
                gram2_ids.append(gram2_id)
            if i < len(word_list) - 2:
                gram3 = "".join(word_list[i:i+3])
                if gram3 not in gram3_dict:
                    gram3 = "[UNK]"
                gram3_id = gram3_dict[gram3]
                gram3_ids.append(gram3_id)
        input_len = sum(input_mask)
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            input_mask = input_mask[:max_seq_length]
        else:
            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            input_ids += [pad_token] * padding_length
            input_mask += [0] * padding_length
        if len(gram2_ids) > max_seq_length:
            gram2_ids = gram2_ids[:max_seq_length]
        else:
            pad2_length = max_seq_length - len(gram2_ids)
            gram2_ids += [pad_token] * pad2_length
        if len(gram3_ids) > max_seq_length:
            gram3_ids = gram3_ids[:max_seq_length]
        else:
            pad3_length = max_seq_length - len(gram3_ids)
            gram3_ids += [pad_token] * pad3_length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(gram2_ids) == max_seq_length
        assert len(gram3_ids) == max_seq_length
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in word_list]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("label_id: %s", example.label)
        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask,
                                      input_len=input_len, label_idx=label_id,
                                      gram2_ids=gram2_ids, gram3_ids=gram3_ids))
    return features


def convert_examples_to_features(examples, max_seq_length, label2id, pad_token=0,
                                 vocab_dict=None):
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        word_list = example.text_a.split()
        input_ids = [vocab_dict.get(word, 1) for word in word_list if word.strip()]
        label_id = label2id[example.label]
        input_mask = [1] * len(input_ids)
        input_len = sum(input_mask)
        # Zero-pad up to the sequence length.
        if len(input_ids) >= max_seq_length:
            input_ids = input_ids[:max_seq_length]
            input_mask = input_mask[:max_seq_length]
        else:
            padding_length = max_seq_length - len(input_ids)
            input_ids += [pad_token] * padding_length
            input_mask += [0] * padding_length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in word_list]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("label_id: %s", example.label)
        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask,
                                      input_len=input_len, label_idx=label_id))
    return features


class JDWordsProcessor(DataProcessor, ABC):
    word_type = None
    data_format = "json"

    def __init__(self, data_dir, word_type=True, data_format=None, ch_flag=True):
        if data_format:
            self.data_format = data_format
        self.ch_flag = ch_flag
        self.word_type = word_type
        self.vocab_dict = {"[UNK]": 1, "[PAD]": 0}
        self.gram2_dict = {"[UNK]": 1, "[PAD]": 0}
        self.gram3_dict = {"[UNK]": 1, "[PAD]": 0}
        tmp_label_set = set()
        with open(f"{data_dir}/train.txt", "r", encoding="utf-8") as fr:
            for line in fr:
                if self.data_format == "json":
                    json_data = json.loads(line.strip())
                    word_list = json_data['words'].split()
                    label = json_data['label']
                    tmp_label_set.add(label)
                elif self.data_format == "ltw":
                    tmpdata = line.strip().split('\t')
                    if len(tmpdata) != 2:
                        continue
                    word_list = tmpdata[1].split()
                    tmp_label_set.add(tmpdata[0])
                elif self.data_format == "wtl":
                    tmpdata = line.strip().split('\t')
                    if len(tmpdata) != 2:
                        continue
                    word_list = tmpdata[0].split()
                    tmp_label_set.add(tmpdata[1])
                else:
                    print("error data format...")
                    exit(1)
                for i, word in enumerate(word_list):
                    if word not in self.vocab_dict:
                        self.vocab_dict[word] = len(self.vocab_dict)
                    if i < len(word_list) - 1:
                        gram2 = "".join(word_list[i: i+2])
                        self.gram2_dict[gram2] = len(self.gram2_dict)
                    if i < len(word_list) - 2:
                        gram3 = "".join(word_list[i: i+3])
                        self.gram3_dict[gram3] = len(self.gram3_dict)
        label_file = f"{data_dir}/labels.txt"
        if os.path.exists(label_file):
            self.label_list = []
            with open(label_file, 'r', encoding="utf-8") as fr:
                for line in fr:
                    self.label_list.append(line.strip())
        else:
            self.label_list = list(tmp_label_set)
            with open(label_file, 'w', encoding='utf-8') as fw:
                fw.write("\n".join(self.label_list))
        self.id2label = {idx: label for idx, label in enumerate(self.label_list)}
        self.label2id = {label: idx for idx, label in enumerate(self.label_list)}

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return self.label_list, self.label2id, self.id2label

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if self.data_format == "json":
                json_data = json.loads(line.strip())
                words = json_data['words']
                label = json_data['label']
            elif self.data_format == "ltw":
                tmpdata = line.split("\t")
                if len(tmpdata) != 2:
                    continue
                label, words = tmpdata
            elif self.data_format == "wtl":
                tmpdata = line.split("\t")
                if len(tmpdata) != 2:
                    continue
                words, label = tmpdata
            guid = "%s-%s" % (set_type, i)
            if self.ch_flag:
                text = words.replace(" ", "")
            else:
                text = words
            if self.word_type:
                examples.append(InputExample(guid=guid, text_a=words, label=label))
            else:
                examples.append(InputExample(guid=guid, text_a=text, label=label))
        return examples


cls_processors = {
    "jd": JDWordsProcessor
}

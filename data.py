import os
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
import torch
from collections import Counter
from math import *
import json


class DataGenerator(object):
    def __init__(self, data_path, data_name, batch_size, tokenizer, split, device="cuda", data_format="trec",
                 add_url=False, label_map=None, vocab=None):
        super(DataGenerator, self).__init__()

        if split == "train":
            self.count_vocab = Counter()
            all_train_docs = []
        self.data = []
        self.tokenizer = tokenizer
        self.data_format = data_format
        self.label_map = {} if label_map is None else label_map
        if data_format == "trec":
            self.fa = open(os.path.join(data_path, "{}/{}/a.toks".format(data_name, split)))
            self.fb = open(os.path.join(data_path, "{}/{}/b.toks".format(data_name, split)))
            self.fsim = open(os.path.join(data_path, "{}/{}/sim.txt".format(data_name, split)))
            self.fid = open(os.path.join(data_path, "{}/{}/id.txt".format(data_name, split)))
            self.lengths = []

            if add_url:
                self.furl = open(os.path.join(data_path, "{}/{}/url.txt".format(data_name, split)))
                for a, b, sim, ID, url in zip(self.fa, self.fb, self.fsim, self.fid, self.furl):
                    self.data.append([sim.replace("\n", ""), a.replace("\n", ""), b.replace("\n", ""), \
                                      ID.replace("\n", ""), url.replace("\n", "")])
            else:
                for a, b, sim, ID in zip(self.fa, self.fb, self.fsim, self.fid):
                    self.data.append([sim.replace("\n", ""), a.replace("\n", ""), b.replace("\n", ""), \
                                      ID.replace("\n", "")])
                    self.lengths.append(len(b.replace("\n", "").split()))
            print("{} {}".format(data_name, sum(self.lengths)/len(self.lengths)))
        elif data_format == "ontonote":
            self.f = open(os.path.join(data_path, "{}/{}.char.bmes".format(data_name, split)))
            label, token = [], []
            for l in self.f:
                ls = l.replace("\n", "").split()
                if len(ls) > 1:
                    if ls[1] not in self.label_map:
                        if split == "test" or split == "dev":
                            print("See new label in {} set: {}".format(split, ls[1]))
                        self.label_map[ls[1]] = len(self.label_map)
                    label.append(self.label_map[ls[1]])
                    token.append(ls[0])
                else:
                    if len(token) > 510:
                        print("find one sentence with length {} on {} set".format(len(token), split))
                        token = token[:510]
                        label = label[:510]
                    self.data.append([token, label])
                    label, token = [], []
            if len(label) > 0:
                self.data.append([token, label])

            print("label_map: {}".format(self.label_map))
        elif data_format == "movie":
            self.f = open(os.path.join(data_path, "{}/{}.tsv".format(data_name, split)))
            self.label_map = {} if label_map is None else label_map
            for l in self.f:
                ls = l.replace("\n", "").split("\t")
                rid = ls[0]
                text = ls[2]
                label = ls[3] if len(ls) == 4 else 2
                self.data.append([rid, text, label])
        elif data_format == "glue":
            self.f = open(os.path.join(data_path, "{}/{}.tsv".format(data_name, split)))
            first = True
            self.f.readline()
            for l in self.f:
                ls = l.replace("\n", "").split("\t")
                query = ls[7]
                doc = ls[8]
                if split == "test":
                    label = 0
                else:
                    label = ls[9]
                if first:
                    first = False
                    print("label: {}".format(label))
                    print("query: {}".format(query))
                    print("doc: {}".format(doc))
                self.data.append([label, query, doc])
        elif self.data_format == "regression":
            self.f = open(os.path.join(data_path, "{}/{}_{}.csv".format(data_name, data_name, split)))
            first = True
            for l in self.f:
                ls = l.replace("\n", "").split("\t")
                label = ls[0]
                query = ls[1]
                doc = ls[2]
                if first:
                    first = False
                    print("label: {}".format(label))
                    print("query: {}".format(query))
                    print("doc: {}".format(doc))
                self.data.append([label, query, doc])
        elif self.data_format == "doc2query":
            self.f = open(os.path.join(data_path, "{}/{}.tsv".format(data_name, split)), encoding='utf-8')
            for l in self.f:
                qid, docid, query, document = l.replace("\n", "").split("\t")
                self.data.append([qid, docid, query, document])

        elif self.data_format == "doc":
            self.f = open(os.path.join(data_path, "{}/{}.tsv".format(data_name, split)), encoding='utf-8')
            for l in self.f:
                docid, document = l.replace("\n", "").split("\t")
                self.data.append([docid, document])

        else:
            self.f = open(os.path.join(data_path, "{}/{}_{}.csv".format(data_name, data_name, split)))
            first = True
            for l in self.f:
                ls = l.replace("\n", "").split("\t")
                if len(ls) < 7:
                    print(l)
                label = ls[0]
                query = ls[2]
                doc = ls[3]
                qid = ls[6]
                docid = ls[7]
                if first:
                    first = False
                    print("label: {}".format(label))
                    print("query: {}".format(query))
                    print("doc: {}".format(doc))
                    print("qid: {}".format(qid))
                    print("docid: {}".format(docid))
                self.data.append([label, query, doc, qid, docid])

        np.random.shuffle(self.data)
        self.data_i = 0
        self.data_size = len(self.data)
        self.add_url = add_url
        self.batch_size = batch_size
        self.device = device
        # self.tokenizer = tokenizer
        self.start = True
        # self.stop_words = set(stopwords.words('english')) 
        if vocab is None:
            self.query_to_f = {}
            f = open("data/msmarco/train.tsv")
            count = 0
            for l in f:
                qid, docid, query, doc = l.split("\t")
                count += 1
                for w in word_tokenize(query):
                    w = w.lower()
                    if w not in self.query_to_f:
                        self.query_to_f[w] = 0
                    self.query_to_f[w] += 1
            self.query_to_idf = {w : log(count / self.query_to_f[w]) for w in self.query_to_f}
            # self.query_to_idf = {w : log(len(all_train_docs) / self.query_to_f[w]) if self.query_to_f[w] !=1 else self.query_to_f[w] for w in self.query_to_f}

            self.vocab = map(lambda x: x[0], sorted([(word, self.query_to_f[word]) for word in query_to_f if word not in stopwords \
                                and word not in string.punctuation], key=lambda x: x[1], reverse=True))
            self.token2id = {w: i for i, w in enumerate(self.vocab)}
            with open('vocab.json', 'w') as fp:
                json.dump(self.query_to_idf, fp)
        else:
            self.vocab = vocab
            self.token2id = {w: i for i, w in enumerate(self.vocab)}
        # self.token2id.json = json.load(open("token2id.json"))
        # self.bias = [self.query_to_idf[w] for w in self.all_words]
        self.stop_words = ["a", "an", "and", "are", "as", "at", "be", "but", "by",\
                          "for", "if", "in", "into", "is", "it",\
                          "no", "not", "of", "on", "or", "such",\
                          "that", "the", "their", "then", "there", "these",\
                          "they", "this", "to", "was", "will", "with"] 

    def get_instance(self):
        ret = self.data[self.data_i % self.data_size]
        self.data_i += 1
        return ret

    def epoch_end(self):
        return self.data_i % self.data_size == 0

    def tokenize_index(self, text, pad=True):
        tokenized_text = self.tokenizer.tokenize(text)
        if pad:
            tokenized_text.insert(0, "[CLS]")
            tokenized_text.append("[SEP]")
        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text[:512])
        return indexed_tokens

    def tokenize_two(self, a, b):
        b_index = self.tokenize_index(b)
        tokenized_text_a = ["[CLS]"] + self.tokenizer.tokenize(a) + ["[SEP]"]
        tokenized_text_b = self.tokenizer.tokenize(b)
        tokenized_text_b = tokenized_text_b[:510 - len(tokenized_text_a)]
        tokenized_text_b.append("[SEP]")
        segments_ids = [0] * len(tokenized_text_a) + [1] * len(tokenized_text_b)
        tokenized_text = tokenized_text_a + tokenized_text_b
        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return indexed_tokens, segments_ids

    def query2label(self, q):
        # q_index = np.array(self.tokenize_index(q, pad=False))
        tokenized_text = word_tokenize(q)
        filtered_tokens = [w for w in tokenized_text if not w in self.stop_words]
        q_index = [self.token2id[t.lower()] for t in filtered_tokens if t.lower() in self.token2id]
        if len(q_index) == 0:
            return None
        label = np.zeros(len(self.token2id), dtype=float)
        label[q_index] = 1
        return label

    def labels2tokens(self, i):
        ii = np.where(i == 1)[0]
        return self.ids2tokens(ii)

    def ids2tokens(self, ids):
        return [self.vocab[i] for i in ids]

    def load_batch(self):
        if self.data_format == "ontonote":
            return self.load_batch_seqlabeling()
        else:
            return self.load_batch_classification()

    def load_batch_seqlabeling(self):
        test_batch, mask_batch, label_batch, token_type_ids_batch = [], [], [], []
        while True:
            if not self.start and self.epoch_end():
                self.start = True
                break
            self.start = False
            instance = self.get_instance()
            token, label = instance
            # This line is important. Otherwise self.data will be modified
            label = label[:]
            assert len(token) == len(label)
            label.insert(0, self.label_map["O"])
            label.append(self.label_map["O"])
            token_index = self.tokenize_index(" ".join(token))
            # print(token, token_index, label, len(token_index), len(label))
            assert len(token_index) == len(label)
            segments_ids = [0] * len(token_index)
            test_batch.append(torch.tensor(token_index))
            token_type_ids_batch.append(torch.tensor(segments_ids))
            mask_batch.append(torch.ones(len(token_index)))
            label_batch.append(torch.tensor(label))
            if len(test_batch) >= self.batch_size or self.epoch_end():
                # Convert inputs to PyTorch tensors
                tokens_tensor = torch.nn.utils.rnn.pad_sequence(test_batch, batch_first=True, padding_value=0).to(
                    self.device)
                segments_tensor = torch.nn.utils.rnn.pad_sequence(token_type_ids_batch, batch_first=True,
                                                                  padding_value=0).to(self.device)
                mask_tensor = torch.nn.utils.rnn.pad_sequence(mask_batch, batch_first=True, padding_value=0).to(
                    self.device)
                label_tensor = torch.nn.utils.rnn.pad_sequence(label_batch, batch_first=True,
                                                               padding_value=self.label_map["O"]).to(self.device)
                assert tokens_tensor.shape == segments_tensor.shape
                assert tokens_tensor.shape == mask_tensor.shape
                assert tokens_tensor.shape == label_tensor.shape
                return (tokens_tensor, segments_tensor, mask_tensor, label_tensor)

    def load_batch_classification(self):
        test_batch, token_type_ids_batch, mask_batch, label_batch, qid_batch, docid_batch, bias_batch = [], [], [], [], [], [], []
        while True:
            if not self.start and self.epoch_end():
                self.start = True
                break
            self.start = False
            instance = self.get_instance()
            if self.data_format == "movie":  # single sentence classification
                rid, text, label = instance
                combine_index = self.tokenize_index(text)
                qid_batch.append(int(rid))
                segments_ids = [0] * len(combine_index)
            elif self.data_format == "doc2query":  # single sentence classification
                qid, docid, query, document = instance
                label = self.query2label(query)
                if label is None:
                    continue
                combine_index = self.tokenize_index(document)
                segments_ids = [0] * len(combine_index)
                qid_batch.append(int(qid))
                docid_batch.append(int(docid))
            elif self.data_format == "doc":  # single sentence classification
                docid, document = instance
                combine_index = self.tokenize_index(document)
                segments_ids = [0] * len(combine_index)
                docid_batch.append(int(docid))
            else:  # sentence pair classification
                if self.data_format == "robust04":
                    label, a, b, qid, docid = instance
                    qid = int(qid)
                    docid = int(docid)
                    qid_batch.append(qid)
                    docid_batch.append(docid)
                else:
                    if len(instance) == 5:
                        label, a, b, ID, url = instance
                    elif len(instance) == 4:
                        label, a, b, ID = instance
                    else:
                        label, a, b = instance
                    if self.add_url:
                        b = b + " " + url
                    if len(instance) >= 4:
                        ls = ID.split()
                        if len(ls) > 1:
                            qid, _, docid, _, _, _ = ls
                            docid = int(docid)
                            docid_batch.append(docid)
                        else:
                            qid = ID
                        qid = int(qid)
                        qid_batch.append(qid)
                combine_index, segments_ids = self.tokenize_two(a, b)
            test_batch.append(torch.tensor(combine_index))
            token_type_ids_batch.append(torch.tensor(segments_ids))
            mask_batch.append(torch.ones(len(combine_index)))
            if self.data_format == "glue" or self.data_format == "regression":
                label_batch.append(float(label))
            elif self.data_format == "doc2query":
                label_batch.append(label)
            elif self.data_format == "doc":
                pass
            else:
                label_batch.append(int(label))
            if len(test_batch) >= self.batch_size or self.epoch_end():
                # Convert inputs to PyTorch tensors
                tokens_tensor = torch.nn.utils.rnn.pad_sequence(test_batch, batch_first=True, padding_value=0).to(
                    self.device)
                segments_tensor = torch.nn.utils.rnn.pad_sequence(token_type_ids_batch, batch_first=True,
                                                                  padding_value=0).to(self.device)
                mask_tensor = torch.nn.utils.rnn.pad_sequence(mask_batch, batch_first=True, padding_value=0).to(
                    self.device)
                if self.data_format == "doc":
                    docid_tensor = torch.tensor(docid_batch, device=self.device)
                    return (tokens_tensor, segments_tensor, mask_tensor, docid_tensor)                 
                else:
                    label_tensor = torch.tensor(label_batch, device=self.device)
                    if len(qid_batch) > 0:
                        qid_tensor = torch.tensor(qid_batch, device=self.device)
                        if len(docid_batch) > 0:
                            docid_tensor = torch.tensor(docid_batch, device=self.device)
                            return (tokens_tensor, segments_tensor, mask_tensor, label_tensor, qid_tensor, docid_tensor)
                        return (tokens_tensor, segments_tensor, mask_tensor, label_tensor, qid_tensor)
                    else:
                        return (tokens_tensor, segments_tensor, mask_tensor, label_tensor)


        return None

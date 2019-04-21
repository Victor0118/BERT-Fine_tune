import copy

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel,BertConfig

class BertMSE(nn.Module):

    def __init__(self, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = float(dropout)

        n_feat = 768

        self.classifier = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(n_feat, 1)
        )


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, label_tensor=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        preds = self.classifier(pooled_output)
        #preds = torch.clamp(preds, 0, 5)
        batch_size = input_ids.size(0)

        if label_tensor is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(preds.view(batch_size), label_tensor)
            return loss
        else:
            return preds


class BertForDoc2Query(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = self.config.vocab_size
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, vocab_size, hidden_size=768, hidden_dropout_prob=0.1):
        config = BertConfig.from_dict({"vocab_size": vocab_size, "hidden_size": hidden_size, "hidden_dropout_prob": hidden_dropout_prob})
        super(BertForDoc2Query, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.num_labels = vocab_size
        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        sigmoid_outputs = torch.sigmoid(logits)
        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            loss_fn = torch.nn.BCELoss()
            labels = labels.float()
            loss = loss_fn(sigmoid_outputs.view(-1, self.num_labels), labels)
            return loss
        else:
            return logits

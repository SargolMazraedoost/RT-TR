#!/usr/bin/env python
"""
@File: net.py
@Author: Sargol Mazraedoost
@Created: 2025/02/10
@Contact: sargol@pukyong.ac.kr
"""

from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize Classification head
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            self.dropout,
            nn.Linear(config.hidden_size // 2, config.num_labels),
        )

    # defalut forward from huggingface
    def forward(self, features, **kwargs):
        x = features.mean(dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(input_dim, 1))
        nn.init.xavier_uniform_(self.attention_weights.data)
        self.tanh = nn.Tanh()

    def forward(self, token_embeddings, attention_mask):
        # Compute attention scores
        attention_scores = torch.matmul(token_embeddings, self.attention_weights).squeeze(-1)
        attention_scores = self.tanh(attention_scores)
        attention_scores = attention_scores.masked_fill(attention_mask == 0, float("-inf"))

        # Convert scores to probabilities
        attention_probs = torch.softmax(attention_scores, dim=1)

        # Expand attention probabilities to match the token embeddings dimension
        attention_probs_expanded = attention_probs.unsqueeze(-1).expand(token_embeddings.size())

        # Compute the weighted sum of token embeddings
        weighted_sum = torch.sum(token_embeddings * attention_probs_expanded, dim=1)

        return weighted_sum


class LSTMClassificationHead(nn.Module):
    """Model with LSTM and a classification head for sentence-level classification tasks."""

    def __init__(
        self,
        num_labels=1,
        dropout=0.1,
        hidden_dim=768,
        lstm_hidden_size=128,
        lstm_layers=1,
        bidirectional=True,
        feature_method="first_token",
        **kwargs,
    ):
        super().__init__()

        self.feature_method = feature_method
        self.hidden_dim = hidden_dim
        self.skip = kwargs.get("skip", False)
        self.cls_skip = kwargs.get("cls_skip", False)

        self.lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size

        if feature_method == "mean_variance":
            self.lstm_output_size = self.lstm_output_size * 2

        if feature_method == "attention":
            self.attention_pooling = AttentionPooling(self.lstm_output_size)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden_size,
            dropout=dropout,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        #

        activation_function = nn.GELU
        dim = self.lstm_output_size
        # Initialize Classification head
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            activation_function(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_labels),
        )

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def attention(self, outputs, hidden, mask=None):
        hidden = hidden.unsqueeze(2)

        attn_weights = torch.bmm(outputs, hidden)

        attn_weights = attn_weights.squeeze(2)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e10)

        soft_attn_weights = F.softmax(attn_weights, dim=1)

        soft_attn_weights = soft_attn_weights.unsqueeze(2)

        weighted = torch.bmm(outputs.transpose(1, 2), soft_attn_weights)

        weighted = weighted.squeeze(2)

        return weighted, soft_attn_weights.squeeze(2)

    def forward(self, out_base, meta=None, attention_mask=None, **kwargs):
        features = out_base[:, :, :]

        features, (hidden, cn) = self.lstm(features)
        # lstm_out shape: (batch_size, seq_len, lstm_output_size)

        if self.feature_method == "mean":
            x = self.mean_pooling(features, attention_mask)
        elif self.feature_method == "variance":
            x = features.var(dim=1)
        elif self.feature_method == "mean_variance":
            x = torch.cat((features.mean(dim=1), features.var(dim=1)), dim=1)
        elif self.feature_method == "last_token":
            x = features[:, -1, :]

            if self.lstm.bidirectional:
                x = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        elif self.feature_method == "attention":
            x = self.attention_pooling(features, attention_mask)
        else:
            x = out_base[:, 0, :]
            # x=torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)  # CLS token
        x = self.head(x)

        return x


class RobertaForRegression(RobertaPreTrainedModel):
    def __init__(
        self,
        config: RobertaConfig,
        last_use=True,
        layers_to_use: Union[int, List[int]] = -1,
        weighted=False,
        use_lora=False,
        lora_config=None,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.last_use = last_use
        self.layers_to_use = layers_to_use if isinstance(layers_to_use, list) else [layers_to_use]
        self.weighted = weighted

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        self.post_init()

        # Apply LORA if specified
        if use_lora:
            if lora_config is None:
                lora_config = LoraConfig(
                    r=64,
                    lora_alpha=128,
                    target_modules=["query", "key", "value", "dense"],
                    lora_dropout=0.1,
                    bias="none",
                    # task_type="SEQ_CLS"
                )
            self.roberta = get_peft_model(self.roberta, lora_config)
            print("Applied LORA to the model.")
            self.roberta.print_trainable_parameters()

        if self.weighted:
            num_layers = len(self.layers_to_use)
            self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)  # Initialize with uniform weights
            self.softmax = nn.Softmax(dim=0)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        return_dict=True,
        output_attentions=False,
        weight=None,
        token_type_ids=None,
        meta_features=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )

        if self.last_use:
            features = outputs.last_hidden_state
        elif self.weighted:
            features = torch.stack([outputs.hidden_states[i] for i in self.layers_to_use], dim=0)
            norm_weights = self.softmax(self.weights)
            features = torch.einsum("l,lbsd->bsd", norm_weights, features)
        else:
            features = torch.cat([outputs.hidden_states[i] for i in self.layers_to_use], dim=-1)

        logits = self.classifier(features, attention_mask=attention_mask)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,  # needed for visualization otherwise comment out
            attentions=outputs.attentions,  # needed for visualization otherwise comment out
        )

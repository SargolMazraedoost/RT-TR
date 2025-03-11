#!/usr/bin/env python
"""
@File: custom_trainer.py
@Author: Sargol Mazraedoost
@Created: 2025/02/14
@Contact: sargol@pukyong.ac.kr
"""

import torch.nn as nn
from transformers import Trainer


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the loss for a given model and inputs.

        Args:
            model (nn.Module): The model to use for computing the loss.
            inputs (dict): A dictionary containing the inputs and labels.
            return_outputs (bool, optional): If True, return the outputs and loss. Defaults to False.

        Returns:
            tuple or float: If return_outputs is True, returns a tuple of (loss, outputs). Otherwise, returns the loss.
        """
        labels = inputs.pop("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute loss

        loss_fct = nn.L1Loss()

        loss = loss_fct(logits.view(-1), labels.float().view(-1))  # , weights.view(-1))

        return (loss, outputs) if return_outputs else loss

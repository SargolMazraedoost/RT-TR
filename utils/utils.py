#!/usr/bin/env python
"""
@File: utils.py
@Author: Sargol Mazraedoost
@Created: 2025/02/14
@Contact: sargol@pukyong.ac.kr
"""

import math
import random
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import SVG, display
from matplotlib.colors import LinearSegmentedColormap
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from torch.optim.lr_scheduler import ExponentialLR
from transformers import (
    AdamW,
)

warnings.simplefilter("ignore")


def seed_torch(seed=1):
    """
    Set the random seed for reproducibility in PyTorch, NumPy, and random.

    Args:
        seed (int): The seed value to set. Defaults to 1.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def find_learning_rate(
    model, train_dataloader, criterion=None, start_lr=1e-8, end_lr=0.01, num_iter=1000, smooth_f=0.05, diverge_th=5
):
    """
    Finds the optimal learning rate for a model using the Learning Rate Finder algorithm.

    The algorithm iterates over the training set, increasing the learning rate exponentially
    from `start_lr` to `end_lr` over `num_iter` iterations. After each iteration, the loss
    is computed and smoothed using an exponential moving average. The algorithm stops when
    the smoothed loss has increased by a factor of `diverge_th` compared to the best loss
    observed so far.

    The algorithm returns three lists: the learning rates used, the losses observed, and
    the smoothed losses observed.

    Args:
        model (nn.Module): The model to optimize.
        train_dataloader (DataLoader): The training data loader.
        criterion (nn.Module, optional): The loss function to use. Defaults to nn.L1Loss.
        start_lr (float, optional): The starting learning rate. Defaults to 1e-8.
        end_lr (float, optional): The ending learning rate. Defaults to 0.01.
        num_iter (int, optional): The number of iterations to run. Defaults to 1000.
        smooth_f (float, optional): The smoothing factor for the exponential moving average.
            Defaults to 0.05.
        diverge_th (int, optional): The threshold for loss divergence. Defaults to 5.

    Returns:
        List[float], List[float], List[float]: The learning rates, losses, and smoothed losses
            observed during the optimization process.
    """
    model.train()
    total_batches = len(train_dataloader)
    num_iter = min(num_iter, total_batches)

    # Use ExponentialLR for smoother lr increase
    optimizer = AdamW(model.parameters(), lr=start_lr)
    scheduler = ExponentialLR(optimizer, math.exp(math.log(end_lr / start_lr) / num_iter))

    if criterion is None:
        criterion = torch.nn.L1Loss()

    lr_list, loss_list, smooth_loss_list = [], [], []
    best_loss = float("inf")

    for step, batch in enumerate(train_dataloader):
        if step >= num_iter:
            break

        batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else criterion(outputs, batch["labels"])

        # Compute smoothed loss
        if step == 0:
            smooth_loss = loss.item()
        else:
            smooth_loss = smooth_loss * (1 - smooth_f) + loss.item() * smooth_f

        if smooth_loss > diverge_th * best_loss:
            print(f"Stopping early, loss has diverged at step {step}")
            break

        loss.backward()
        optimizer.step()
        scheduler.step()

        lr = scheduler.get_last_lr()[0]

        lr_list.append(lr)
        loss_list.append(loss.item())
        smooth_loss_list.append(smooth_loss)

        if smooth_loss < best_loss:
            best_loss = smooth_loss

        if step % 10 == 0 or step == num_iter - 1:
            print(f"Step {step}: loss={loss.item():.4f}, smooth_loss={smooth_loss:.4f}, lr={lr:.7f}")
    # save data
    data = {"lr": lr_list, "loss": loss_list, "smooth_loss": smooth_loss_list}
    pd.DataFrame(data).to_csv("lr_finder_data.csv", index=False)
    return lr_list, loss_list, smooth_loss_list


def collate_fn(batch):
    collated = {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }
    if "weight" in batch[0]:
        collated["weight"] = torch.stack([item["weight"] for item in batch])
    return collated


def get_attention_maps(model, tokenizer, smiles_string, scaler, test_dataset, layers=None):
    idx = smiles_string
    model.eval()
    with torch.no_grad():
        inputs = test_dataset[idx]
        # get converted tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"])
        # remove padding tokens
        tokens = [t for t in tokens if t != tokenizer.pad_token]
        outputs = model(
            input_ids=inputs["input_ids"].unsqueeze(0),
            attention_mask=inputs["attention_mask"].unsqueeze(0),
            output_attentions=True,
            labels=inputs["labels"].unsqueeze(0),
        )

        input_unscale = scaler.inverse_transform(np.array(inputs["labels"]).reshape(-1, 1))[0][0]
        result_unscale = scaler.inverse_transform(np.array(outputs.logits).reshape(-1, 1))[0][0]

    attentions = outputs.attentions
    avg_attention = torch.mean(torch.cat(attentions), dim=(1,)).detach().numpy()
    # remove padding attention weights
    # avg_attention = avg_attention[:, : len(tokens), : len(tokens)]

    # Assuming avg_attention has shape (6, 35, 35)
    if layers is not None:
        layers_to_average = avg_attention[layers, : len(tokens), : len(tokens)]
        avg_attention = np.mean(layers_to_average, axis=0)
    else:
        avg_attention = avg_attention[5, : len(tokens), : len(tokens)]

    return avg_attention, tokens, attentions, input_unscale, result_unscale


def process_atoms(molecule):
    """
    Extracts atom indices and symbols from the molecule.

    Args:
        molecule: RDKit molecule object.

    Returns:
        tuple: Lists of atom indices and symbols.
    """
    atom_indices = [atom.GetIdx() for atom in molecule.GetAtoms()]
    atom_symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    return atom_indices, atom_symbols


def validate_tokens(tokens, molecule):
    """
    Validates that the tokens list matches the SMILES string of the molecule.

    Args:
        tokens (list): List of SMILES tokens.
        molecule: RDKit molecule object.

    Returns:
        None: Raises an error if tokens do not match the molecule.
    """
    smiles_from_tokens = "".join(tokens)
    canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
    if smiles_from_tokens != canonical_smiles:
        raise ValueError(
            f"Mismatch between tokens and molecule:\nTokens: {smiles_from_tokens}\nMolecule SMILES: {canonical_smiles}"
        )


def calculate_averaged_scores(tokens, scores, atom_symbols):
    """
    Calculates averaged scores for atoms, considering digits in tokens.

    Args:
        tokens (list): List of SMILES tokens.
        scores (list): Scores corresponding to each token.
        atom_symbols (list): List of atom symbols.

    Returns:
        list: Averaged scores for each atom.
    """
    averaged_scores = []
    token_index = 0
    atom_symbols = list(map(str.upper, atom_symbols))
    # print(atom_symbols)
    for atom_symbol in atom_symbols:
        # print(atom_symbol, tokens[token_index].upper())
        # Ensure token_index does not exceed the bounds of tokens
        while token_index < len(tokens) and tokens[token_index].upper() != atom_symbol:
            token_index += 1

        # If token_index is out of bounds, raise an error
        if token_index >= len(tokens):
            raise ValueError(f"Atom symbol '{atom_symbol}' not found in tokens.")

        # If the next token is a digit, average the current score with the next one
        if token_index + 1 < len(tokens) and tokens[token_index + 1].isdigit():
            averaged_scores.append((scores[token_index] + scores[token_index + 1]) / 2)
            token_index += 2
        else:
            averaged_scores.append(scores[token_index])
            token_index += 1

    return averaged_scores


def normalize_scores(scores):
    """
    Normalizes scores to the range [0, 1].

    Args:
        scores (list): List of scores.

    Returns:
        list: Normalized scores.
    """
    max_score = max(scores)
    return [float(score) / max_score for score in scores]


def expand_tokens_and_scores(tokens, scores):
    """
    Expand tokens and scores arrays by splitting multi-character non-atomic tokens
    while keeping recognized two-character atomic symbols intact.
    Args:
        tokens (list): List of tokens
        scores (list): List of scores corresponding to tokens
    Returns:
        expanded_tokens (list): Expanded list of tokens
        expanded_scores (list): Expanded list of scores
    """
    two_char_atoms = {
        "Li",
        "Na",
        "Al",
        "Si",
        "Cl",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        # Added more two-character atoms
    }

    expanded_tokens = []
    expanded_scores = []

    i = 0
    while i < len(tokens):
        token = tokens[i]
        score = scores[i]

        if len(token) == 2 and token in two_char_atoms:
            expanded_tokens.append(token)
            expanded_scores.append(score)
            i += 1  # Move to the next token
            continue

        elif len(token) > 1:
            j = 0
            while j < len(token):
                if j + 1 < len(token) and token[j : j + 2] in two_char_atoms:
                    expanded_tokens.append(token[j : j + 2])
                    expanded_scores.append(score)
                    j += 2
                else:
                    expanded_tokens.append(token[j])
                    expanded_scores.append(score)
                    j += 1
        else:
            expanded_tokens.append(token)
            expanded_scores.append(score)

        i += 1

    return expanded_tokens, expanded_scores

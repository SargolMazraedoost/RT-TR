#!/usr/bin/env python
"""
@File: dataset.py
@Author: Sargol Mazraedoost
@Created: 2025/02/14
@Contact: sargol@pukyong.ac.kr
"""

import random
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from torch.utils.data import Dataset


class SMILESDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        smiles_column: str,
        value_column: str,
        tokenizer,
        max_length: int,
        weight_column: Optional[str] = None,
        augment: bool = True,
        aug_prob: float = 0.5,
        add_special_tokens: bool = True,
        aug_types: List[str] = ["enumerate", "stereochem", "conformer", "fragment"],
        n_jobs: int = -1,
        num_meta=None,
        labels_task2=None,
        cache_size: int = 1000,
        rt_noise_std: float = 0.04,
        **kwargs,
    ):
        """
        Initialize the SMILESDataset.

        Args:
            dataframe (pd.DataFrame): Pandas DataFrame containing the data.
            smiles_column (str): Name of the column containing the SMILES strings.
            value_column (str): Name of the column containing the values.
            tokenizer: Tokenizer to be used for tokenizing the SMILES strings.
            max_length (int): Maximum length of the SMILES strings.
            weight_column (Optional[str], optional): Name of the column containing the weights. Defaults to None.
            augment (bool, optional): Whether to apply data augmentation. Defaults to True.
            aug_prob (float, optional): Probability of applying data augmentation. Defaults to 0.5.
            add_special_tokens (bool, optional): Whether to add special tokens to the SMILES strings. Defaults to True.
            aug_types (List[str], optional): List of data augmentation types. Defaults to ["enumerate", "stereochem", "conformer", "fragment"].
            n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.
            num_meta (int, optional): Number of meta features to use. Defaults to None.
            labels_task2 (str, optional): Name of the column containing the labels for task 2. Defaults to None.
            cache_size (int, optional): Size of the cache for the augmentation function. Defaults to 1000.
            rt_noise_std (float, optional): Standard deviation of the noise to add to the retention times. Defaults to 0.04.
        """
        self.data = dataframe
        self.smiles_column = smiles_column
        self.value_column = value_column
        self.weight_column = weight_column
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.aug_prob = aug_prob
        self.aug_types = aug_types
        self.num_meta = num_meta
        self.labels_task2 = labels_task2
        self.add_special_tokens = add_special_tokens
        self.n_jobs = n_jobs if n_jobs > 0 else None
        self.augmentation_history: List[Tuple[int, str]] = []
        self.rt_noise_std = rt_noise_std

        # Identify meta feature columns
        if self.num_meta is not None:
            self.meta_columns = [
                col
                for col in self.data.columns
                if col not in {self.smiles_column, self.value_column, self.weight_column}
            ][: self.num_meta]

        # Initialize the caching mechanism
        self.augment_smiles = lru_cache(maxsize=cache_size)(self._augment_smiles)

    def _convert_to_float(self, value):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.0  # or some other default value
        else:
            return 0.0  # or some other default value

    def __len__(self):
        return len(self.data)

    def _augment_smiles(self, smiles: str) -> Tuple[str, str]:
        if not self.augment or random.random() > self.aug_prob:
            return smiles, "none"

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, "invalid"  # Return original if invalid

        augmentation_type = random.choice(self.aug_types)

        if augmentation_type == "enumerate":
            return Chem.MolToSmiles(mol, doRandom=True), "enumerate"
        if augmentation_type == "stereochem":
            stereo_atoms = [
                atom.GetIdx() for atom in mol.GetAtoms() if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
            ]
            if stereo_atoms:
                new_mol = Chem.Mol(mol)
                atom_idx = random.choice(stereo_atoms)
                new_mol.GetAtomWithIdx(atom_idx).InvertChirality()
                return Chem.MolToSmiles(new_mol), "stereochem"
        elif augmentation_type == "conformer":
            conformer_mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(conformer_mol, randomSeed=random.randint(1, 100000))
            AllChem.MMFFOptimizeMolecule(conformer_mol)
            return Chem.MolToSmiles(conformer_mol), "conformer"
        elif augmentation_type == "fragment":
            # Simple fragment replacement (replace -OH with -NH2)
            patt = Chem.MolFromSmarts("[OH]")
            repl = Chem.MolFromSmarts("[NH2]")
            new_mol = AllChem.ReplaceSubstructs(mol, patt, repl, replaceAll=True)[0]
            return Chem.MolToSmiles(new_mol), "fragment"

        return smiles, "none"  # Return original if no augmentation was applied

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        original_smiles = row[self.smiles_column]

        augmented_smiles, aug_type = self.augment_smiles(original_smiles)
        self.augmentation_history.append((idx, aug_type))

        value = row[self.value_column]

        # Add Gaussian noise to the RT value if augmentation was applied

        # if aug_type != 'none':
        #     value += np.random.normal(0, self.rt_noise_std)

        encoding = self.tokenizer(
            augmented_smiles,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(value, dtype=torch.float),
        }

        # Add weight if weight_column is specified
        if self.weight_column:
            weight = row[self.weight_column]
            item["weight"] = torch.tensor(weight, dtype=torch.float)

        # Add meta features for future use
        if self.num_meta is not None:
            meta_features = [self._convert_to_float(row[col]) for col in self.meta_columns]
            item["meta_features"] = torch.tensor(meta_features, dtype=torch.float)

        # Add labels for task 2 if specified for multitask learning (e.g., regression)
        if self.labels_task2 is not None:
            labels_task2 = row[self.labels_task2]
            item["labels_task2"] = torch.tensor(labels_task2, dtype=torch.float)

        return item

    def get_batch(self, indices: List[int]) -> List[Dict[str, torch.Tensor]]:
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            return list(executor.map(self.__getitem__, indices))

    def draw_molecule(self, idx: int, size: Tuple[int, int] = (300, 300)) -> Image.Image:
        """Draw the molecule at the given index."""
        smiles = self.data.iloc[idx][self.smiles_column]
        return draw_molecule(smiles, size)

    def draw_augmented_molecule(self, idx: int, size: Tuple[int, int] = (300, 300)) -> Image.Image:
        """Draw the augmented molecule at the given index."""
        original_smiles = self.data.iloc[idx][self.smiles_column]
        augmented_smiles, _ = self.augment_smiles(original_smiles)
        return draw_molecule(augmented_smiles, size)

    def get_augmentation_stats(self) -> Dict[str, int]:
        """Get statistics on applied augmentations."""
        return {
            aug_type: count
            for aug_type, count in pd.DataFrame(self.augmentation_history, columns=["idx", "aug_type"])["aug_type"]
            .value_counts()
            .items()
        }


def draw_molecule(smiles: str, size: Tuple[int, int] = (300, 300)) -> Image.Image:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)

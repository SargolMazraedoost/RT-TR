import warnings
from functools import partial

import pandas as pd
import torch
from net.net import LSTMClassificationHead, RobertaForRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from utils.dataset import SMILESDataset
from utils.metrics import compute_metrics
from utils.plotting import plot_lr_finder
from utils.utils import find_learning_rate, seed_torch

warnings.simplefilter("ignore")
# %% Set seed
seed = 1
seed_torch(seed)
# %% Load model
save_model_path = "./models/ChemBERTa-zinc-base-v1"
tokenizer = AutoTokenizer.from_pretrained(save_model_path)
config = AutoConfig.from_pretrained(save_model_path)

model = RobertaForRegression.from_pretrained(
    save_model_path,
    num_labels=1,
    ignore_mismatched_sizes=True,
    last_use=False,
    layers_to_use=[-1, -3, -5],
    use_lora=False,
)

model.classifier = LSTMClassificationHead(
    hidden_dim=768 * 3,
    lstm_hidden_size=768,
    num_labels=1,
    feature_method="mean",
    lstm_layers=1,
    dropout=0.1,
    bidirectional=True,
    skip=False,
    cls_skip=False,
    out_conv=None,
)

# %% Load data
df = pd.read_csv("./data/SMRT_dataset.txt", sep="\t")

df_n, df_val = train_test_split(df, test_size=0.1, random_state=seed)
df_train, df_test = train_test_split(df_n, test_size=0.1, random_state=seed)
df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)
numeric_columns = ["rt"]

scaler = PowerTransformer(method="box-cox", standardize=True)

df_train[numeric_columns] = scaler.fit_transform(df_train[numeric_columns])
df_test[numeric_columns] = scaler.transform(df_test[numeric_columns])

compute_metrics = partial(compute_metrics, scaler=scaler)

train_dataset = SMILESDataset(
    df_train,
    "smiles",
    "rt",
    tokenizer,
    max_length=89,
    augment=True,
    aug_prob=0.6,
    aug_types=["enumerate"],
    add_special_tokens=False,
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# %% Find learning rate
lrs, losses, smooth_losses = find_learning_rate(model, train_dataloader)
plot_lr_finder(lrs, losses, smooth_losses)

# %%
import argparse
import os
import warnings
from argparse import Namespace
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import torch
from net.net import LSTMClassificationHead, RobertaForRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    PowerTransformer,
)
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
)
from utils.custom_trainer import CustomTrainer
from utils.dataset import SMILESDataset
from utils.metrics import compute_metrics
from utils.plotting import process_predictions
from utils.utils import seed_torch

warnings.simplefilter("ignore")

# %% Arguments and seed
parser = argparse.ArgumentParser(description="Train a model ....")
parser.add_argument("--model_name", type=str, default="roberta", help="model name")
parser.add_argument("--path", type=str, default="./_runs/", help="path to the output directory")

args = parser.parse_args()

seed = 1
seed_torch(seed)

# %% Load the model
save_model_path = "./models/ChemBERTa-zinc-base-v1"
tokenizer = AutoTokenizer.from_pretrained(save_model_path)
config = AutoConfig.from_pretrained(save_model_path)

dim = config.hidden_size

model = RobertaForRegression.from_pretrained(
    save_model_path,
    num_labels=1,
    ignore_mismatched_sizes=True,
    last_use=False,
    layers_to_use=[-1, -3, -5],
)

model.classifier = LSTMClassificationHead(
    hidden_dim=dim if model.last_use else dim * len(model.layers_to_use),
    lstm_hidden_size=dim,
    num_labels=1,
    feature_method="mean",
    lstm_layers=1,
    dropout=0.1,
    bidirectional=True,
)

# path_safe_tensors= f"./test/_runs/2024_07_24/20_23_testevl/testevl/checkpoint-353346/model.safetensors"
# model.load_state_dict(load_file(path_safe_tensors))

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params:,}")

# %% Load the Dataset

# Load the dataset and split it into train, validation, and test sets
df = pd.read_csv("./data/SMRT_dataset.txt", sep="\t")
df_train, df_test = train_test_split(df, test_size=0.1, random_state=seed)
df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=seed)

# reset index
df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

# save datasets
data_path = os.makedirs(f"{args.path}/data", exist_ok=True)
df_train.to_csv(f"{args.path}/data/train.csv")
df_val.to_csv(f"{args.path}/data/val.csv")
df_test.to_csv(f"{args.path}/data/test.csv")

numeric_columns = ["rt"]

# handle the skewed data distribution and scale the data
scaler = PowerTransformer(method="box-cox", standardize=True)
# scaler = StandardScaler()

df_train[numeric_columns] = scaler.fit_transform(df_train[numeric_columns])
df_test[numeric_columns] = scaler.transform(df_test[numeric_columns])
df_val[numeric_columns] = scaler.transform(df_val[numeric_columns])

compute_metrics = partial(compute_metrics, scaler=scaler)

# %% build the dataset class for training and evaluation

max_length = 89
train_dataset = SMILESDataset(
    df_train,
    "smiles",
    "rt",
    tokenizer,
    max_length=max_length,
    augment=True,
    aug_prob=0.6,
    aug_types=["enumerate"],
    add_special_tokens=False,
)

test_dataset = SMILESDataset(
    df_test,
    "smiles",
    "rt",
    tokenizer,
    max_length=max_length,
    augment=False,
    add_special_tokens=False,
)

eval_dataset = SMILESDataset(
    df_val,
    "smiles",
    "rt",
    tokenizer,
    max_length=max_length,
    augment=False,
    add_special_tokens=False,
)

# %% Train the model

# Training arguments
num_epochs = 200
learning_rate = 5e-5
batch_size = 32
gradient_accumulation = 1
scheduler = "reduce_lr_on_plateau"
wd = 0.1
gradnorm = 1.0
scheduler_kwargs = {"patience": 5, "threshold": 5e-4, "factor": 0.2, "eps": 1e-9, "min_lr": 1e-7}

train_args = TrainingArguments(
    f"{args.path}/{args.model_name}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    load_best_model_at_end=True,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size * 4,
    gradient_accumulation_steps=gradient_accumulation,
    lr_scheduler_type=scheduler,
    lr_scheduler_kwargs=scheduler_kwargs,
    num_train_epochs=num_epochs,
    max_grad_norm=gradnorm,
    weight_decay=wd,
    seed=seed,
    warmup_ratio=0.1,
    use_cpu=False,
    remove_unused_columns=False,
)

# check device
print(train_args.device)

# Initialize the trainer
trainer = CustomTrainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
history = trainer.train()

# Get augmentation statistics
aug_stats = train_dataset.get_augmentation_stats()
print("Augmentation statistics:", aug_stats)

# %% Plot the training and validation losses
validation_losses = [item["eval_loss"] for item in trainer.state.log_history if "eval_loss" in item]
train_losses = [item["loss"] for item in trainer.state.log_history if "loss" in item]
num_epochs = len(train_losses)

plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), validation_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Losses")
plt.legend()
plt.savefig(f"{args.path}/train_val_losses_prediction_{args.model_name}.png")
plt.close()

# %% Evaluate on both eval and test sets
results = trainer.evaluate(eval_dataset=eval_dataset)
print("Evaluation results:", results)

test_results = trainer.evaluate(eval_dataset=test_dataset)
print("Test results:", test_results)
print("_" * 50)
# %% Load the saved model

best_model_path = trainer.state.best_model_checkpoint
print("best_model_path:", best_model_path)

print("Testing the model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Lists to store true and predicted values
y_true_list = []
y_pred_list = []

test_loader = trainer.get_eval_dataloader(test_dataset)
model.eval()
with torch.no_grad():
    for batch in test_loader:
        # # Push numpy to CUDA tensors
        batch.to(device)

        labels = batch.get("labels")

        # Forward pass
        outputs = model(**batch).logits
        # Append true and predicted values
        # Assuming a regression model
        y_true_list.extend(labels.cpu().numpy())
        y_pred_list.extend(outputs.cpu().numpy())

# Process the predictions
args = Namespace(path="./figures", model_name=args.model_name)

process_predictions(y_true_list, y_pred_list, scaler, args)

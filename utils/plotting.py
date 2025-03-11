#!/usr/bin/env python
"""
@File: plotting.py
@Author: Sargol Mazraedoost
@Created: 2025/02/10
@Contact: sargol@pukyong.ac.kr
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import torch
from IPython.display import SVG, display
from matplotlib.colors import LinearSegmentedColormap
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from .utils import calculate_averaged_scores, normalize_scores, process_atoms


def process_predictions(y_true_list, y_pred_list, scaler, args):
    # Convert lists to NumPy arrays
    """
    Process the predictions and actual values to calculate evaluation metrics, create plots, and save results.

    Args:
        y_true_list (list): List of actual retention times.
        y_pred_list (list): List of predicted retention times.
        scaler (sklearn.preprocessing.TransformerMixin): The scaler used to transform the data.
        args (Namespace): The parsed arguments.

    Returns:
        tuple: R2 score, mean absolute error, mean absolute percentage error, and root mean squared error.
    """
    y_true = np.array(y_true_list).reshape(-1, 1)
    y_pred = np.array(y_pred_list).reshape(-1, 1)

    # Save the initial results
    pd.DataFrame({"y_true": y_true.flatten(), "y_pred": y_pred.flatten()}).to_csv(
        f"{args.path}/prediction_{args.model_name}.csv", index=False
    )

    # Scale back to original rt
    y_true = scaler.inverse_transform(y_true).reshape(-1)
    y_pred = scaler.inverse_transform(y_pred).reshape(-1)

    # Turn y_pred and y_true into DataFrames
    result = pd.DataFrame({"rt": y_true, "prediction": y_pred})

    # Save the scaled result
    result.to_csv(f"{args.path}/results_{args.model_name}.csv", index=False)

    # Calculate evaluation metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print("r2 score:", r2)
    print("rmse:", rmse)
    print("MAE:", mae)
    print("MAPE:", mape)

    # Calculate the distance of each point from the diagonal line
    distances = np.abs(result["rt"] - result["prediction"])

    # Create a colormap based on the distances
    cmap = plt.cm.get_cmap("viridis")
    colors = cmap(distances / distances.max())

    # Create the scatter plot and error distribution plot
    fig, axs = plt.subplots(2, 2, figsize=(16, 14))

    # Scatter plot of actual vs predicted
    axs[0, 0].scatter(result["rt"], result["prediction"], alpha=0.5, c=colors, edgecolors="k", s=50)
    axs[0, 0].plot(result["rt"].values, result["rt"].values, color="red", lw=2)
    axs[0, 0].set_xlabel("Actual")
    axs[0, 0].set_ylabel("Predicted")
    axs[0, 0].set_title("Actual vs Predicted")
    axs[0, 0].set_xlim(450, 1500)
    axs[0, 0].set_xlim(450, 1500)
    # Error distribution plot
    error = result["rt"] - result["prediction"]
    axs[0, 1].hist(error, bins=100, edgecolor="black", color="dodgerblue")
    axs[0, 1].set_xlabel("Prediction Error (secs)")
    axs[0, 1].set_ylabel("Count")
    axs[0, 1].set_title("Prediction Error Distribution")
    axs[0, 1].set_xlim(-700, 700)

    # # Residuals vs. Fitted Values Plot
    # axs[1, 0].scatter(result["prediction"], error, alpha=0.5, edgecolors="k", s=50, c="dodgerblue")
    # axs[1, 0].axhline(0, color="red", lw=2)
    # axs[1, 0].set_xlabel("Fitted Values")
    # axs[1, 0].set_ylabel("Residuals")
    # axs[1, 0].set_title("Residuals vs. Fitted Values")

    # # QQ Plot
    # sm.qqplot(error, line="s", ax=axs[1, 1])
    # axs[1, 1].set_title("QQ Plot")

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{args.path}/prediction_{args.model_name}.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [y_true, y_pred],
        labels=["Actual", "Predicted"],
        showmeans=True,
        meanline=True,
        notch=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
    )
    plt.title("Boxplot of Actual vs Predicted")
    plt.ylabel("Retention Time (secs)")
    plt.grid(True)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{args.path}/boxplot_{args.model_name}.png")
    plt.close()

    return r2, mae, mape, rmse


def plot_lr_finder(lrs, losses, smooth_losses, skip_start=5, skip_end=5):
    """
    Plots the learning rate finder results.

    Args:
        lrs (list or array-like): List of learning rates tested.
        losses (list or array-like): Corresponding loss values for each learning rate.
        smooth_losses (list or array-like): Smoothed loss values for better visualization.
        skip_start (int, optional): Number of initial points to skip in the plot to avoid noise. Default is 5.
        skip_end (int, optional): Number of final points to skip in the plot to avoid noise. Default is 5.

    Displays a plot of the raw and smoothed losses against the learning rates on a logarithmic scale
    to help identify the optimal learning rate.
    """

    plt.figure(figsize=(10, 6))
    plt.plot(
        lrs[skip_start:-skip_end],
        losses[skip_start:-skip_end],
        label="Raw Loss",
        color="slateblue",
    )
    plt.plot(
        lrs[skip_start:-skip_end], smooth_losses[skip_start:-skip_end], label="Smoothed Loss", color="yellow", lw=2
    )
    plt.xscale("log")
    # plt.yscale('log')
    plt.xlabel("Learning Rate", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Learning Rate Finder")
    plt.legend()
    plt.savefig("/data/home/hadi/RT-TR/figures/lr_finder.png")
    plt.show()
    plt.close()


def map_scores_to_colors(molecule, scores, colors):
    """
    Maps scores to colors for atoms and bonds.

    Args:
        molecule: RDKit molecule object.
        scores (list): Normalized scores for atoms.
        colors (list): List of hex color codes.

    Returns:
        tuple: Highlight dictionaries for atoms and bonds.
    """

    def get_color(score, colors):
        n_colors = len(colors)
        idx = int(score * (n_colors - 1))  # Scale the score to an index
        return mcolors.hex2color(colors[idx])

    # Highlight atoms
    highlight_atoms = {i: get_color(score, colors) for i, score in enumerate(scores)}

    # Highlight bonds by averaging the scores of the two atoms involved
    highlight_bonds = {}
    for bond in molecule.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        avg_score = (scores[start_idx] + scores[end_idx]) / 2.0
        highlight_bonds[bond.GetIdx()] = get_color(avg_score, colors)

    return highlight_atoms, highlight_bonds


def draw_molecule_with_highlights(molecule, highlight_atoms, highlight_bonds, tokens):
    """
    Draws the molecule with highlights using RDKit.

    Args:
        molecule: RDKit molecule object.
        highlight_atoms (dict): Dictionary of atom indices and colors.
        highlight_bonds (dict): Dictionary of bond indices and colors.
        tokens (list): List of SMILES tokens.

    Returns:
        str: SVG representation of the molecule.
    """
    d2d = rdMolDraw2D.MolDraw2DSVG(400, 300)
    dopts = d2d.drawOptions()
    dopts.useBWAtomPalette()  # Use black-and-white palette for atoms
    dopts.addAtomIndices = True  # Add atom indices for clarity

    d2d.DrawMolecule(
        molecule,
        highlightAtoms=list(highlight_atoms.keys()),
        highlightBonds=list(highlight_bonds.keys()),
        highlightAtomColors=highlight_atoms,
        highlightBondColors=highlight_bonds,
        legend=f"SMILES: {''.join(tokens)}",
    )
    d2d.FinishDrawing()
    return d2d.GetDrawingText()


def display_colorbar(colors):
    """
    Displays a horizontal colorbar.

    Args:
        colors (list): List of hex color codes.

    Returns:
        None: Displays the colorbar.
    """
    fig, ax = plt.subplots(figsize=(3, 0.2))
    fig.subplots_adjust(bottom=0.5)

    # Convert hex colors to RGB for the colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("heatmap", colors)
    norm = plt.Normalize(0, 1)

    # Create the color bar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="horizontal",
        ticks=np.linspace(0, 1, len(colors)),
    )
    cbar.ax.set_xticklabels(["Low", "", "", "", "", "", "High"])  # Customize labels if needed
    plt.show()


def visualize_attention(attention_tensor):
    """
    Visualize attention weights across layers, heads, and tokens.

    Args:
    - attention_tensor: torch.Tensor of shape [num_layers, num_heads, num_tokens, num_tokens]
    """
    # Ensure the tensor is numpy for visualization
    if torch.is_tensor(attention_tensor):
        attention_tensor = attention_tensor.detach().cpu().numpy()

    num_layers, num_heads, num_tokens, _ = attention_tensor.shape

    # Create a grid of subplots
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(20, 15), sharex=True, sharey=True)
    fig.suptitle("Attention Weights Across Layers and Heads", fontsize=16)

    # Iterate through layers and heads
    for layer in range(num_layers):
        for head in range(num_heads):
            # Get attention weights for this specific layer and head
            head_weights = attention_tensor[layer, head]

            # Plot heatmap
            sns.heatmap(head_weights, ax=axes[layer, head], cmap="Reds", cbar=False, square=True)

            axes[layer, head].set_title(f"Layer {layer + 1}, Head {head + 1}")
            # axes[layer, head].set_xlabel('Token Index')
            # axes[layer, head].set_ylabel('Token Index')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_attention_heatmap(smiles_string, attention_weights, tokens):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(attention_weights.T, cmap="Reds", ax=ax, cbar=False)
    ax.set_xticks(np.arange(len(tokens)) + 0.5)
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(tokens)) + 0.5)
    ax.set_yticklabels(tokens, rotation=0)
    ax.set_xlabel("SMILES Token")
    ax.set_ylabel("SMILES Token")
    ax.set_title("Attention Heatmap for SMILES String")

    # make borders of heatmap square black
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.5)

    plt.colorbar(ax.collections[0], ax=ax, label="Attention Weight", pad=0.02, aspect=50)
    plt.tight_layout()

    return fig


def plot_attention_on_sequence(
    smiles_string, attention_weights, tokens, actual, predicted, base_dir, save=True, axis=0
):
    # Aggregate attention weights
    agg_attention = attention_weights.mean(axis=axis)
    # Normalize attention weights
    norm_attention = (agg_attention - agg_attention.min()) / (agg_attention.max() - agg_attention.min())

    # Create a custom red colormap
    # colors = ["#FFFFFF", "#FFCCCC", "#FF9999", "#FF6666", "#FF3333", "#FF0000", "#CC0000"]
    # n_bins = 50
    # cmap = LinearSegmentedColormap.from_list("custom_red", colors, N=n_bins)

    # cmap= plt.cm.get_cmap('Reds', 20)

    cmap = plt.cm.magma_r

    # Extract a subset of the colormap (from 0 to 0.5, which is the middle)
    colors = cmap(np.linspace(0, 0.7, 200))  # Adjust the resolution as needed 20 /10
    cmap_half = LinearSegmentedColormap.from_list("Reds_half", colors, 100)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(30, 2))
    ax.set_xlim(0, len(tokens))
    ax.set_ylim(0, 1)

    # Plot each character with its corresponding attention

    spacing = 1  # Adjust this value to control spacing between tokens
    for i, (char, attention) in enumerate(zip(tokens, norm_attention)):
        color = cmap_half(attention)
        if i == 0:
            i = -0.5
        ax.text(
            i * spacing,
            0.5,
            char,
            ha="center",
            va="center",
            fontsize=28,
            bbox=dict(facecolor=color, edgecolor="none", pad=0.1, alpha=0.7),
            fontfamily="Arial",
        )

    # Remove axes
    ax.axis("off")

    plt.title(
        f"Attention Visualization on SMILES Sequence\n\n \
              SMILES: {''.join(tokens)} \n\n \
              Actual RT: {actual:.2f}   |   Predicted RT: {predicted:.2f}\n ",
        fontsize=28,
    )

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_half, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.0, shrink=1.5)
    cbar.ax.set_position([0.1, 0.1, 0.8, 0.1])
    # cbar xticks font size
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label("Normalized Attention Score", fontsize=24)

    # save figure
    if save:
        plt.savefig(f"{base_dir}/{smiles_string}_attention.png", bbox_inches="tight", dpi=600)

    plt.tight_layout()
    return fig


def visualize_molecule_with_colorbar(tokens, scores, colors=None):
    """
    Visualizes a molecule with a colorbar based on atom and bond scores.

    Args:
        tokens (list): List of SMILES tokens representing the molecule.
        scores (list): Scores corresponding to each token.
        colors (list or None): Custom colormap. If None, uses a default colormap.

    Returns:
        None: Displays the molecule visualization with a colorbar.
    """
    # Step 1: Validate inputs
    if len(tokens) != len(scores):
        raise ValueError("Tokens and scores must have the same length.")

    # Step 2: Define default colormap if not provided
    if colors is None:
        cmap = plt.cm.magma_r
        colors = [mcolors.to_hex(cmap(i)) for i in np.linspace(0, 0.7, 100)]

    # Step 3: Convert tokens to SMILES string and create molecule object
    smiles = "".join(tokens)
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES string generated from tokens.")

    # Step 4: Process atom indices and scores
    atom_indices, atom_symbols = process_atoms(molecule)
    averaged_scores = calculate_averaged_scores(tokens, scores, atom_symbols)

    # Normalize scores between 0 and 1
    normalized_scores = normalize_scores(averaged_scores)

    # Step 5: Map scores to colors
    highlight_atoms, highlight_bonds = map_scores_to_colors(molecule, normalized_scores, colors)

    # Step 6: Visualize the molecule using RDKit
    svg = draw_molecule_with_highlights(molecule, highlight_atoms, highlight_bonds, tokens)

    # Step 7: Display the SVG and colorbar
    display(SVG(svg))
    # display_colorbar(colors)

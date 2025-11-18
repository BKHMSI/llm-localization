import os 
import argparse
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

from model_utils import model_name_map, get_num_blocks, get_hidden_dim

CACHE_DIR = os.environ.get("LOC_CACHE", f"cache")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=False, default="Llama-3.2-3B-Instruct", help="Model name")
    parser.add_argument("--network", type=str, required=False, default="language", help="Network to consider")
    parser.add_argument("--percentage", type=float, required=False, default=1.0, help="Percentage of units to consider")
    parser.add_argument("--threshold", type=float, required=False, default=0.05, help="p-value threshold for selectivity")
    args = parser.parse_args()

    percentage = args.percentage
    threshold = args.threshold
    network = args.network
    model_name = args.model_name

    plot_data = {"selectivity": [], "layer_num": [], "model_name": []}  

    model_name = os.path.basename(model_name)
    num_layers = get_num_blocks(model_name)
    hidden_dim = get_hidden_dim(model_name)

    print(f"Model: {model_name}")

    pooling = "mean" if network != "language" else "last-token"
    model_loc_path = f"{model_name}_network={network}_pooling={pooling}_range=100-100_perc={percentage}_nunits=None_pretrained=True.npy"

    cache_dir = "cache"
    lang_mask_path = f"{cache_dir}/{model_loc_path}"
    if not os.path.exists(lang_mask_path):
        raise ValueError(f"Path does not exist: {lang_mask_path}")

    lang_mask = np.load(lang_mask_path)

    for i in range(num_layers):
        layer_mask = lang_mask[i]

        value = 0 if len(layer_mask) == 0 else (layer_mask.sum() / hidden_dim) * 100
        plot_data["selectivity"].append(value) 

        plot_data["layer_num"].append((i+1))
        plot_data["model_name"].append(model_name_map[model_name])

    df = pd.DataFrame(plot_data)
    pivot_df = df.pivot_table(index="layer_num", columns="model_name", values="selectivity", aggfunc='mean')

    plt.figure(figsize=(4, 6))
    sns.set_theme(context="paper", font_scale=2, style="white")

    ax = sns.heatmap(pivot_df, cmap="viridis", annot=True, fmt=".1f", cbar=False)
    for t in ax.texts: 
        t.set_fontsize(10)
        t.set_text(t.get_text() + " %")

    sns.despine()
    plt.xlabel("")
    plt.ylabel("")
    plt.title("")
    plt.tight_layout()

    plt.savefig(f"{CACHE_DIR}/heatmap_model={model_name}_network={network}.png")
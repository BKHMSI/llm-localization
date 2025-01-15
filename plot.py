import os 
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

from model_utils import model_name_map, get_num_blocks, get_hidden_dim

percentage = 1
threshold = 0.05
network = "multiple-demand" #"theory-of-mind"
model_name = "Llama-3.2-3B-Instruct"

plot_data = {"selectivity": [], "layer_num": [], "model_name": []}  

model_name = os.path.basename(model_name)
num_layers = get_num_blocks(model_name)
hidden_dim = get_hidden_dim(model_name)

print(f"Model: {model_name}")

pooling = "mean" if network != "lang" else "last-token"
model_loc_path = f"{model_name}_network={network}_pooling={pooling}_range=100-100_perc={percentage}_nunits=None_pretrained=True.npy"
model_loc_path = f"{model_name}_network={network}_pooling={pooling}_pretrained=True_pvalues.npy"

cache_dir = "cache"
lang_mask_path = f"{cache_dir}/{model_loc_path}"
if not os.path.exists(lang_mask_path):
    raise ValueError(f"Path does not exist: {lang_mask_path}")

lang_mask_p_values = np.load(lang_mask_path)
lang_mask = lang_mask_p_values < threshold

for i in range(num_layers):
    values = 1 - lang_mask_p_values[i][lang_mask[i]]

    value = 0 if len(values) == 0 else (values.sum() / hidden_dim) * 100
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

plt.savefig(f"cache/heatmap_model={model_name}_network={network}.png")
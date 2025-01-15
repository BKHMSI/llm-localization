<h1> 
  The LLM Language Network 
  <br>
  A Neuroscientific Approach for Identifying Causally Task-Relevant Units
</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2411.02280-b31b1b.svg)](https://arxiv.org/abs/2411.02280)

**Authors**: Badr AlKhamissi, Greta Tuckute, Antoine Bosselut*, Martin Schrimpf*
<p> * Equal Supervision </p>


<div align='center'>
<img style="width: 75%" src='assets/llm-localization.png'></img>
</div>

## Setup

1. Create conda environment: `conda create -n llm-loc python=3.10`
2. Activate environment: `conda activate llm-loc`
3. Install packages: `pip install -r requirements.txt`

### Repository Structure

#### Root Directory
- **`localize.py`**:  
  The main script for localizing selective units within a specified model.
  
- **`datasets.py`**:  
  Handles localizer stimuli loading and processing.

- **`utils.py`**:  
  Contains utility functions used throughout the repository.

- **`model_utils.py`**:  
  Provides model-related utilities.

- **`generate_lesion.py`**:  
  Generates text after applying lesions (ablations) to the localized units.

#### Folders

- **`stimuli/`**:  
  Contains the localizer stimuli required for identifying selective units. These scripts to download the stimuli can be found in `scripts`.

- **`scripts/`**:  
  Contains bash scripts, including `download_lang_stimuli.sh` and `download_tom_stimuli.sh`, which is used to download the localizer stimuli into the `stimuli` folder, as well as example scripts to run localization.

- **`cache/`**:  
  Stores the localized units' masks, which can be reused or overwritten based on the script options.

## Arguments Description

- `--model-name` (`str`, **required**):  
  Specifies the HuggingFace model name that will be localized. This is a required argument.  
  **Example**: `--model-name meta-llama/Llama-3.2-1B`

- `--percentage` (`float`, *optional*):  
  Defines the percentage of units to localize.  
  **Example**: `--percentage 1` (localizes 1% of the units)

- `--localize-range` (`str`, *optional*, default: `"100-100"`):  
  Determines the percentile range of units to localize.  
  - `"100-100"`: Localizes the top-selective units (most selective).  
  - `"0-0"`: Localizes the least-selective units.  
  - `"x-y"`: Localize random units within that percentile range where `y > x`.

  **Example**: `--localize-range 80-90`

- `--network` (`str`, *optional*, default: `"language"`, choices: `["language", "theory-of-mind", "multiple-demand]`):  
  Specifies the network type to localize.  
  **Example**: `--network language`

- `--pooling` (`str`, *optional*, default: `"last-token"`, choices: `["last-token", "mean"]`):  
  Defines the method for token aggregation when localizing units:  
  - `"last-token"`: Uses the last token of the sequence.  
  - `"mean"`: Averages over all tokens in the sequence.  
  **Example**: `--pooling mean`

- `--num-units` (`int`, *optional*):  
  Specifies the exact number of units to localize. If `--percentage` is provided, it overrides this argument.  
  **Example**: `--num-units 1024`

- `--seed` (`int`, *optional*, default: `42`):  
  Sets the random seed for reproducibility of results.  
  **Example**: `--seed 123`

- `--device` (`str`, *optional*):  
  Specifies the device to use (e.g., `"cpu"`, `"cuda:0"`). If not provided, the device will be automatically selected.  
  **Example**: `--device cuda:0`

- `--untrained` (`flag`, *optional*):  
  Use this flag to localize units on an untrained version of the model. No arguments needed for this flag.  
  **Example**: `--untrained` (applies to untrained models)

- `--overwrite` (`flag`, *optional*):  
  Use this flag to overwrite an existing mask if a cached version is found.  
  **Example**: `--overwrite`

## Usage Instructions

1. **Download Localizer Stimuli:**
   Run the following command to download the required language stimuli:
   ```bash
   bash scripts/download_lang_stimuli.sh
   bash scripts/download_tom_stimuli.sh
   ```
   The files will be saved in the `stimuli` directory.

2. **Run the Localization Script:**
   Use the following command to localize the top language-selective units for your chosen model:
   ```bash
   python localize.py 
        --model-name <model_name> 
        --percentage <percentage>
        --network <network> 
        --localize-range 100-100 
        --pooling last-token
   ```
   This will create a mask in the `cache` directory, which can be used to ablate or extract the identified network-selective units. See the `scripts` folder for examples. 

3. **Run the Generation with Ablation Script:**
   See the example in `scripts/generate_lesion.sh`

## Abstract

Large language models (LLMs) exhibit remarkable capabilities on not just language tasks, but also various tasks that are not linguistic in nature, such as logical reasoning and social inference. In the human brain, neuroscience has identified a core language system that selectively and causally supports language processing. We here ask whether similar specialization for language emerges in LLMs. We identify language-selective units within 18 popular LLMs, using the same localization approach that is used in neuroscience. We then establish the causal role of these units by demonstrating that ablating LLM language-selective units -- but not random units -- leads to drastic deficits in language tasks. Correspondingly, language-selective LLM units are more aligned to brain recordings from the human language system than random units. Finally, we investigate whether our localization method extends to other cognitive domains: while we find specialized networks in some LLMs for reasoning and social capabilities, there are substantial differences among models. These findings provide functional and causal evidence for specialization in large language models, and highlight parallels with the functional organization in the brain.

## Citation 
```
@article{alkhamissi2024-llm-localization,
      title={The LLM Language Network: A Neuroscientific Approach for Identifying Causally Task-Relevant Units}, 
      author={Badr AlKhamissi and Greta Tuckute and Antoine Bosselut and Martin Schrimpf},
      year={2024},
      eprint={2411.02280},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.02280}, 
}
```

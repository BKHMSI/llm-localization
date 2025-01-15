import numpy as np
import pandas as pd
from glob import glob
from torch.utils.data import Dataset

from utils import read_story, read_question

class MDLocDataset(Dataset):
    def __init__(self):  
        num_examples = 100

        self.positive = []
        np.random.seed(42)
        for idx in range(num_examples):
            num_1 = np.random.randint(100, 200)
            num_2 = np.random.randint(100, 200)
            add_or_subtract = np.random.choice(["+", "-"])
            if add_or_subtract == "+":
                question = f"Solve {num_1} + {num_2}?"
                answer = num_1 + num_2
            else:
                question = f"Solve {num_1} - {num_2}?"
                answer = num_1 - num_2
            self.positive.append(f"Question: {question}\nAnswer: {answer}")

        self.negative = []
        np.random.seed(42)
        for idx in range(num_examples):
            num_1 = np.random.randint(1, 20)
            num_2 = np.random.randint(1, 20)
            add_or_subtract = np.random.choice(["+", "-"])
            if add_or_subtract == "+":
                question = f"Solve {num_1} + {num_2}?"
                answer = num_1 + num_2
            else:
                question = f"Solve {num_1} - {num_2}?"
                answer = num_1 - num_2
            self.negative.append(f"Question: {question}\nAnswer: {answer}")

    def __getitem__(self, idx):
        return self.positive[idx].strip(), self.negative[idx].strip()
        
    def __len__(self):
        return len(self.positive) 

class LangLocDataset(Dataset):
    def __init__(self):
        dirpath = "stimuli/language"
        paths = glob(f"{dirpath}/*.csv")
        vocab = set()

        data = pd.read_csv(paths[0])
        for path in paths[1:]:
            run_data = pd.read_csv(path)
            data = pd.concat([data, run_data])

        data["sent"] = data["stim2"].apply(str.lower)

        vocab.update(data["stim2"].apply(str.lower).tolist())
        for stimuli_idx in range(3, 14):
            data["sent"] += " " + data[f"stim{stimuli_idx}"].apply(str.lower)
            vocab.update(data[f"stim{stimuli_idx}"].apply(str.lower).tolist())

        self.vocab = sorted(list(vocab))
        self.w2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2w = {i: w for i, w in enumerate(self.vocab)}

        self.positive = data[data["stim14"]=="S"]["sent"]
        self.negative = data[data["stim14"]=="N"]["sent"]

    def __getitem__(self, idx):
        return self.positive.iloc[idx].strip(), self.negative.iloc[idx].strip()
        
    def __len__(self):
        return len(self.positive)

class TOMLocDataset(Dataset):
    def __init__(self):
        dirpath = "stimuli/tom/tomloc"
        instruction = "In this experiment, you will read a series of sentences and then answer True/False questions about them. Press button 1 to answer 'true' and button 2 to answer 'false'."
        context_template = "{instruction}\nStory: {story}\nQuestion: {question}\nAnswer: {answer}"
        belief_stories = [read_story(f"{dirpath}/{idx}b_story.txt") for idx in range(1, 11)]
        photograph_stories = [read_story(f"{dirpath}/{idx}p_story.txt") for idx in range(1, 11)]

        belief_question = [read_question(f"{dirpath}/{idx}b_question.txt") for idx in range(1, 11)]
        photograph_question = [read_question(f"{dirpath}/{idx}p_question.txt") for idx in range(1, 11)]

        self.positive = [context_template.format(instruction=instruction, story=story, question=question, answer=np.random.choice(["True", "False"])) for story, question in zip(belief_stories, belief_question)]
        self.negative = [context_template.format(instruction=instruction, story=story, question=question, answer=np.random.choice(["True", "False"])) for story, question in zip(photograph_stories, photograph_question)]

    def __getitem__(self, idx):
        return self.positive[idx].strip(), self.negative[idx].strip()
    
    def __len__(self):
        return len(self.positive)

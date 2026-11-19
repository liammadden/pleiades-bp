"""
Download a portion of the TinyStories dataset (https://arxiv.org/abs/2305.07759),
tokenize it, and split it into training and test data sets.
"""

from dataclasses import dataclass
from datasets import load_dataset
from itertools import count
from nltk.tokenize import wordpunct_tokenize
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import CustomTextDataset

@dataclass
class Data:
    vocab_size: int = 0
    seq_length: int = 0
    n: int = 100 # number of samples, perhaps batch_size * niter
    training_dataset: any = None
    test_dataset: any = None

    def get_data(self):
        full_dataset = self.process_data()
        self.vocab_size, self.training_dataset, self.test_dataset = (
            self.tokenize_data(full_dataset)
        )

    def process_data(self):
        np.random.seed(0)  # change numpy seed here
        # Load 1% of 2.1 million story dataset.
        full_dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")[
            "text"
        ]
        full_dataset = np.array(full_dataset)
        sample = np.random.permutation(np.size(full_dataset))[0 : 2 * self.n]
        full_dataset = full_dataset[sample]
        full_dataset = full_dataset.tolist()
        return full_dataset

    def tokenize_data(self, full_dataset):
        # Tokenize data.
        datasetTokens = []
        j = 0
        for _, story in enumerate(full_dataset):
            tokenized_story = wordpunct_tokenize(story)
            if len(tokenized_story) >= self.seq_length:
                tokenized_story = tokenized_story[: self.seq_length]
                datasetTokens.append(tokenized_story)
                j = j + 1
                if j == 2 * self.n:
                    break

        # Create vocab dictionary.
        token2id = {}
        id_gen = count(0)
        for story in datasetTokens:
            for word in story:
                if word not in token2id:
                    new_id = next(id_gen)
                    token2id[word] = new_id
        vocab_size = len(token2id)

        # Convert tokens to IDs.
        datasetIDs = []
        for story in datasetTokens:
            storyIDs = []
            for word in story:
                storyIDs.append(token2id[word])
            datasetIDs.append(storyIDs)
        datasetIDs = torch.tensor(datasetIDs)

        # Split into training and test data sets.
        tokenized_dataset = CustomTextDataset(sequence=datasetIDs)
        training_dataset = tokenized_dataset[0 : self.n]
        test_dataset = tokenized_dataset[self.n : 2 * self.n]

        return vocab_size, training_dataset, test_dataset

# Convert data to Python Dataset.
class CustomTextDataset(Dataset):
    def __init__(self, sequence):
        self.sequence = sequence

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        sequence = self.sequence[idx]
        return sequence
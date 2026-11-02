"""Data split, tokenization, Dataset, and DataLoaders for next-token prediction in Assignment 3."""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CONTEXT_LENGTH = 256
END_OF_TEXT = "<|endoftext|>"


def split_stories(text_path, data_dir, val_fraction=0.05):
    """Split the stories into training and validation sets, and save them.

    args:
        text_path: path to the TinyStories text file
        data_dir: directory in which the two text files are saved
        val_fraction: fraction of the stories used for validation
    """
    ####################################
    # Your implementation starts here. #
    ####################################
    with open(text_path, encoding="utf-8") as f:
        sample_text = f.read()
    stories = sample_text.split("<|endoftext|>")
    stories = [story.strip() for story in stories if story.strip()]
    num_val = int(len(stories) * val_fraction)
    val=stories[:num_val]
    train=stories[num_val:]
    with open(os.path.join(data_dir, "train.txt"),"w", encoding="utf-8") as train_file:
        for story in train:
            train_file.write(story+ "\n")
            train_file.write("<|endoftext|>"+ "\n")
    with open(os.path.join(data_dir, "val.txt"),"w", encoding="utf-8") as val_file:
        for story in val:
            val_file.write(story+ "\n")
            val_file.write("<|endoftext|>"+ "\n")
    ##################################
    # Your implementation ends here. #
    ##################################


def tokenize_stories(text_path, tokenizer, tokens_path):
    """Convert all stories in the text file into token IDs and save them.

    args:
        text_path: path to a text file of stories
        tokenizer: a tokenizers.Tokenizer
        tokens_path: path of the .npy file to save
    """
    ####################################
    # Your implementation starts here. #
    ####################################
    with open(text_path, encoding="utf-8") as f:
        text = f.read()
    stories = [story.strip() for story in text.split(END_OF_TEXT) if story.strip()]
    end_of_text_id = tokenizer.token_to_id(END_OF_TEXT)

    # Encode batches to avoid keeping every story's Encoding object in memory.
    token_batches = []
    batch_size = 10_000
    for start in range(0, len(stories), batch_size):
        encodings = tokenizer.encode_batch(stories[start:start + batch_size])
        batch_ids = []
        for encoding in encodings:
            batch_ids.extend(encoding.ids)
            batch_ids.append(end_of_text_id)
        token_batches.append(np.array(batch_ids, dtype=np.uint16))

    tokens = np.concatenate(token_batches) if token_batches else np.empty(0, dtype=np.uint16)
    np.save(tokens_path, tokens)
    ##################################
    # Your implementation ends here. #
    ##################################


class TinyStoriesDataset(Dataset):
    """Return (input, target) windows of a token sequence."""

    def __init__(self, tokens, context_length=CONTEXT_LENGTH):
        self.tokens = tokens
        self.context_length = context_length

    def __len__(self):
        """Return the number of non-overlapping windows."""
        num_windows = None
        ####################################
        # Your implementation starts here. #
        ####################################
        # Each window needs one extra token for the last next-token target.
        num_windows = max(0, (len(self.tokens) - 1) // self.context_length)
        ##################################
        # Your implementation ends here. #
        ##################################
        return num_windows

    def __getitem__(self, index):
        """Return the input and target tokens of window `index`.

        outputs:
            x: shape (context_length,), torch.long, the input tokens
            y: shape (context_length,), torch.long, the target tokens, i.e.,
               the input tokens shifted left by one position
        """
        x, y = None, None
        ####################################
        # Your implementation starts here. #
        ####################################
        start = index * self.context_length
        end = start + self.context_length
        x = torch.tensor(self.tokens[start:end], dtype=torch.long)
        y = torch.tensor(self.tokens[start + 1:end + 1], dtype=torch.long)
        ##################################
        # Your implementation ends here. #
        ##################################
        return x, y


def get_loader(data_dir, split, batch_size, context_length=CONTEXT_LENGTH):
    """Create the DataLoader of one set.

    args:
        data_dir: directory containing train.npy and val.npy
        split: "train" or "val"
        batch_size: number of windows per batch
        context_length: number of input tokens in each window
    outputs:
        loader: a DataLoader that returns batches (x, y), each of shape
            (batch_size, context_length)
    """
    loader = None
    ####################################
    # Your implementation starts here. #
    ####################################
    tokens = np.load(os.path.join(data_dir, f"{split}.npy"))
    dataset = TinyStoriesDataset(tokens, context_length=context_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
    ##################################
    # Your implementation ends here. #
    ##################################
    return loader

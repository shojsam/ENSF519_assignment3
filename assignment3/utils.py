"""Provided helper for downloading TinyStories."""

import urllib.request

DATASET_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
NUM_BYTES = 500_000_000  # first 500 MB of the 2.2 GB training file
END_OF_TEXT = b"<|endoftext|>"


def download_tinystories(text_path):
    """Download the first 500 MB of the TinyStories training file.

    The file is cut after the last complete story, so that it ends with
    "<|endoftext|>".
    """
    print("Downloading TinyStories ...")
    request = urllib.request.Request(DATASET_URL, headers={"Range": f"bytes=0-{NUM_BYTES - 1}"})
    with urllib.request.urlopen(request) as response:
        data = response.read()
    end = data.rfind(END_OF_TEXT) + len(END_OF_TEXT)
    with open(text_path, "wb") as f:
        f.write(data[:end])
    print("Saved to", text_path)

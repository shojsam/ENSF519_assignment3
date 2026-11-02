# ENSF 519 Assignment 3 — A Small Language Model for Short Stories

## Setup

Notebooks 00 and 01 run on a CPU, but a GPU is recommended for training the model in notebook 02. You can use your own computer, ideally with an NVIDIA GPU (CUDA) or Apple Silicon (MPS), [Google Colab](https://colab.research.google.com/) with a GPU runtime, or TALC. TALC is optional and only has 15 GPUs for the whole course, so do not wait until the last few days to use it. We do not administer TALC, so we cannot help with problems on the cluster.

**Option 1**: Local environment using `uv`

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), for example on macOS with `curl -LsSf https://astral.sh/uv/install.sh | sh`.
2. From the assignment directory, create the virtual environment in `.venv/`:

   - with an NVIDIA GPU: `uv sync --extra cuda` (PyTorch for CUDA 12.6). You only need a recent NVIDIA driver; the CUDA toolkit does not need to be installed separately.
   - on Apple Silicon or without a GPU: `uv sync --extra cpu` (on Apple Silicon, this version of PyTorch can also use the GPU through MPS)

   For other setups, such as an AMD GPU with ROCm (Linux only), see [PyTorch&#39;s installation guide](https://pytorch.org/get-started/locally/).
3. Start Jupyter from the assignment directory and run the notebooks in order. The file paths in the notebooks are relative to this directory.

**Option 2**: Using TALC

1. Connect to TALC using your UCalgary IT account, and clone the assignment into your home directory:

   ```bash
   ssh <your_IT_account_name>@talc.ucalgary.ca
   git clone --no-local /work/TALC/ensf519.53_2026f/assignment_templates/assignment3.git ENSF519_assignment3
   ```
2. For interactive jobs, it is suggested to use a CPU node while working through the first two notebooks:

   ```bash
   salloc --partition=cpu16 --time=01:00:00 --cpus-per-task=4 --mem=8G
   ```

   Before working through the third notebook, request a GPU node instead:

   ```bash
   salloc --partition=gpu --time=01:00:00 --gpus-per-node=1 --cpus-per-task=2 --mem=8G
   ```
3. On the allocated node, load Python and activate the course environment. On a GPU node, also load the CUDA 12.6.2 toolkit before activating the environment.

   ```bash
   module load python/3.14.6
   module load cuda/12.6.2 # GPU jobs only
   source /work/TALC/ensf519.53_2026f/envs/a3-py314/bin/activate
   cd ENSF519_assignment3
   jupyter lab --no-browser --ip=0.0.0.0 --port=8888
   ```

   Omit the `module load cuda/12.6.2` line for a CPU job.

For more details about remote editing, requesting computing resources, and connecting to JupyterLab through port forwarding, refer to the TALC guide published earlier in the course.

## Assignment overview

In this assignment, you will build and train a small GPT-style language model on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), a dataset of short stories written with simple words. Text is split into tokens with a byte-pair encoding (BPE) tokenizer of 4,096 tokens, and the model learns to predict the next token from the previous ones.

The structure of this assignment is as follows:

```
ENSF519_assignment3/
├── 00_data_and_tokenization.ipynb
├── 01_transformer_model.ipynb
├── 02_training_and_generation.ipynb
├── assets/
├── assignment3/
│   ├── __init__.py
│   ├── data.py            # *tokenization, data split, Dataset, and DataLoaders
│   ├── model.py           # *Transformer language model
│   ├── training.py        # *loss, training, and evaluation functions
│   └── utils.py           # provided download helper
├── data/
│   ├── example.txt        # three short stories for checking tokenize_stories()
│   └── tokenizer.json     # BPE tokenizer (vocabulary and merge rules)
├── .python-version
├── make_submission.py
├── make_submission.sh
├── pyproject.toml
├── report.md
├── README.md
└── uv.lock
```

In this assignment, you are expected to complete the implementation of the files with stars (*) above, conduct the experiments in `02_training_and_generation.ipynb`, and complete your report in `report.md`.

You are suggested to work through the notebooks from 00 to 02 in order:

1. `00_data_and_tokenization.ipynb` first downloads TinyStories (500 MB) into `data/`. It then introduces the stories and tokenization with byte-pair encoding (BPE). You will convert all stories into token IDs with the provided tokenizer, and implement the data split and the Dataset and DataLoaders for next-token prediction.
2. `01_transformer_model.ipynb` builds the GPT-style model step by step: causal self-attention, multi-head attention, the Transformer block, the full model, and the next-token prediction loss.
3. `02_training_and_generation.ipynb` implements the training and evaluation loops, trains the model, generates stories, and visualizes the attention weights. The report questions refer to sections of this notebook; answer them in `report.md`.

Detailed instructions and code for checking your implementation are included in the notebooks.

## Submission

After completing the code, notebooks, and `report.md`, create the submission ZIP from the assignment directory:

- macOS/Linux: `sh make_submission.sh`
- Windows (PowerShell): `py make_submission.py` (or `python make_submission.py`)

Both commands create `ENSF519_assignment3_submission.zip`. The dataset, virtual environment, and training-output folders are intentionally excluded. Upload this ZIP file to D2L.

## Rubric

| Component                                                   |        Marks |
| ----------------------------------------------------------- | -----------: |
| Data split, tokenization, Dataset, and DataLoaders (Part 0) |            5 |
| Transformer model and loss (Part 1)                         |            5 |
| Training and evaluation loops (Part 2)                      |            2 |
| Report questions                                            |            6 |
| AI use declaration                                          |            1 |
| Reflection                                                  |            1 |
| **Total**                                             | **20** |

Detailed marks for individual methods and questions are provided in the notebooks and [`report.md`](report.md).

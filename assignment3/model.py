"""GPT-style Transformer language model for Assignment 3."""

import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .data import CONTEXT_LENGTH


class AttentionHead(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, d_model, head_dim):
        super().__init__()
        self.query = nn.Linear(d_model, head_dim)
        self.key = nn.Linear(d_model, head_dim)
        self.value = nn.Linear(d_model, head_dim)
        self.attention_weights = None  # set in forward(), used for visualization

    def forward(self, x):
        """Let every position attend to itself and the positions before it.

        args:
            x: shape (batch_size, T, d_model)
        outputs:
            z: shape (batch_size, T, head_dim)
        """
        T = x.shape[1]
        scores, weights, z = None, None, None
        ####################################
        # Your implementation starts here. #
        ####################################
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        ##################################
        # Your implementation ends here. #
        ##################################

        # Causal mask: position t cannot attend to the positions after t.
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        masked_scores = scores.masked_fill(mask, float("-inf"))

        ####################################
        # Your implementation starts here. #
        ####################################
        weights = F.softmax(masked_scores, dim=-1)
        z = weights @ v
        ##################################
        # Your implementation ends here. #
        ##################################
        self.attention_weights = weights.detach()  # used for visualization in Part 2
        return z


class MultiHeadAttention(nn.Module):
    """Several attention heads side by side, followed by an output projection."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        head_dim = d_model // n_heads
        self.heads = nn.ModuleList([AttentionHead(d_model, head_dim) for _ in range(n_heads)])
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        args:
            x: shape (batch_size, T, d_model)
        outputs:
            out: shape (batch_size, T, d_model)
        """
        out = None
        ####################################
        # Your implementation starts here. #
        ####################################
        head_outputs = [head(x) for head in self.heads]
        out = self.projection(torch.cat(head_outputs, dim=-1))
        ##################################
        # Your implementation ends here. #
        ##################################
        return out


class Block(nn.Module):
    """A Transformer block: attention and MLP, each with pre-LayerNorm and a residual connection."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        """
        args:
            x: shape (batch_size, T, d_model)
        outputs:
            x: shape (batch_size, T, d_model)
        """
        ####################################
        # Your implementation starts here. #
        ####################################
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        ##################################
        # Your implementation ends here. #
        ##################################
        return x


class GPT(nn.Module):
    """A GPT-style language model that predicts the next token at every position."""

    def __init__(self, vocab_size, context_length=CONTEXT_LENGTH, d_model=256, n_layers=4, n_heads=4):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        self.blocks = nn.Sequential(*[Block(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids):
        """
        args:
            ids: shape (batch_size, T), torch.long, token IDs with T <= context_length
        outputs:
            logits: shape (batch_size, T, vocab_size), the scores of the next token
                at every position
        """
        logits = None
        ####################################
        # Your implementation starts here. #
        ####################################
        T = ids.shape[1]
        positions = torch.arange(T, device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        x = self.blocks(x)
        logits = self.head(self.norm(x))
        ##################################
        # Your implementation ends here. #
        ##################################
        return logits

    @torch.no_grad()
    def generate(self, ids, num_new_tokens, temperature=1.0, top_k=None):
        """Append num_new_tokens sampled tokens to ids, one token at a time.

        You do not need to modify this function.
        args:
            ids: shape (batch_size, T), torch.long, the prompt
            num_new_tokens: number of tokens to generate
            temperature: the logits are divided by temperature before sampling
            top_k: if not None, sample only from the top_k most likely tokens
        outputs:
            ids: shape (batch_size, T + num_new_tokens), the prompt followed by
                the generated tokens
        """
        for _ in range(num_new_tokens):
            # 1. the raw logits at the last position, (batch_size, vocab_size)
            context = ids[:, -self.context_length:]  # the model reads at most context_length tokens
            logits = self(context)[:, -1, :]
            logits = logits / temperature

            # 2. top-k: keep the top_k largest logits, set the others to -inf
            if top_k is not None:
                kth_largest = torch.topk(logits, top_k).values[:, -1:]
                logits = logits.masked_fill(logits < kth_largest, float("-inf"))

            # 3. sample the next token, (batch_size, 1)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            # 4. append it to ids
            ids = torch.cat([ids, next_id], dim=1)
        return ids

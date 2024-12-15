import torch
import torch.nn as nn
import tiktoken 
from torch.utils.data import Dataset, DataLoader
from modules.norms.layer_norm import LayerNorm
from modules.attention.causal_attention import CausalMultiHeadAttention
from modules.activations.gelu import GELU

#Why mean of 0 and variance of 1 useful? 
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_embedding = nn.Embedding(config["context_length"], config["emb_dim"])
        self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.drop_embedding = nn.Dropout(config["drop_rate"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])]
        )

        self.layer_norm = LayerNorm(config["emb_dim"])
        self.projection_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias = False)

    def forward(self, x):
        batch_size, seq_len = x.shape
        token_embeddings = self.token_embedding(x)
        pos_embeddings = self.pos_embedding(torch.arange(seq_len).to(x.device))
        embeddings = token_embeddings + pos_embeddings
        embeddings = self.drop_embedding(embeddings)
        embeddings = self.transformer_blocks(embeddings)
        embeddings = self.layer_norm(embeddings)
        logits = self.projection_head(embeddings)
        return logits
    

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["emb_dim"], config["emb_dim"] * 4),
             GELU(),
             nn.Linear(config["emb_dim"] * 4, config["emb_dim"]),
        )
    def forward(self, x):
        return self.layers(x)
    

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalMultiHeadAttention(
            d_in = config["emb_dim"],
            d_out = config["emb_dim"],
            context_length = config["context_length"],
            num_heads = config["n_heads"],
            dropout = config["drop_rate"],
            qkv_bias = True
        )
        self.ff = FeedForward(config)
        self.layernorm1 = LayerNorm(config["emb_dim"])
        self.layernorm2 = LayerNorm(config["emb_dim"])
        self.drop_shortcut = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.layernorm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = shortcut + x

        shortcut = x
        x = self.layernorm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = shortcut + x
        return x
    
import torch
import torch.nn as nn

class GPTConfig:
    def __init__(self, vocab_size, block_size=128, n_layer=4, n_head=4, n_embd=64, dropout=0.1):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dropout=config.dropout,
            batch_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layer)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"Input too long ({T} tokens), max is {self.config.block_size}")

        tok_emb = self.token_embedding(idx)            # (B, T, C)
        pos_emb = self.pos_embedding[:, :T, :]         # (1, T, C)
        x = self.dropout(tok_emb + pos_emb)

        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(idx.device)  # (T, T)
        x = self.blocks(x, mask=mask)  # causal mask

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits

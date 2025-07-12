# GPT Model Architecture

## Overview
This is a simplified GPT (Generative Pre-trained Transformer) implementation using PyTorch's built-in TransformerEncoder layers.

## Configuration Parameters
```
GPTConfig:
├── vocab_size: Vocabulary size
├── block_size: 128 (maximum sequence length)
├── n_layer: 4 (number of transformer layers)
├── n_head: 4 (number of attention heads)
├── n_embd: 64 (embedding dimension)
└── dropout: 0.1 (dropout rate)
```

## Model Architecture Flow

```
Input: Token indices (B, T)
    │
    ▼
┌─────────────────────────┐
│   Token Embedding      │
│   (vocab_size → n_embd) │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Positional Embedding  │
│   (1, block_size, n_embd)│
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│      Addition +         │
│    Dropout Layer        │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Causal Attention Mask  │
│   (upper triangular)    │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ TransformerEncoder x4   │
│                         │
│  ┌─────────────────┐   │
│  │ MultiHeadAttn   │   │
│  │ (4 heads)       │   │
│  └─────────────────┘   │
│           │             │
│  ┌─────────────────┐   │
│  │ Feed Forward    │   │
│  │ Network         │   │
│  └─────────────────┘   │
│                         │
│  (Repeated 4 times)     │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Layer Normalization   │
│      (ln_f)             │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│    Output Head          │
│ (n_embd → vocab_size)   │
└─────────────────────────┘
    │
    ▼
Output: Logits (B, T, vocab_size)
```

## Component Details

### 1. Input Processing
- **Token Embedding**: Maps token indices to dense vectors (vocab_size → n_embd)
- **Positional Embedding**: Learned position encodings (block_size, n_embd)
- **Dropout**: Applied to the sum of token and positional embeddings

### 2. Transformer Stack
- **4 TransformerEncoder Layers**: Each containing:
  - Multi-head self-attention (4 heads)
  - Feed-forward network
  - Residual connections
  - Layer normalization
- **Causal Mask**: Upper triangular mask prevents attention to future tokens

### 3. Output Processing
- **Final Layer Norm**: Normalizes the final hidden states
- **Output Head**: Linear layer projecting to vocabulary size for next-token prediction

## Key Features
- **Autoregressive**: Uses causal masking for left-to-right generation
- **Compact**: Small model with 4 layers and 4 attention heads
- **Efficient**: Uses PyTorch's built-in transformer components
- **Configurable**: All hyperparameters defined in GPTConfig

## Data Flow Summary
```
Tokens → Embeddings → Attention Layers → LayerNorm → Logits
   │         │              │              │         │
   │         │              │              │         └─→ Next token probabilities
   │         │              │              └─→ Final hidden states
   │         │              └─→ Contextual representations
   │         └─→ Dense vectors with position info
   └─→ Discrete token indices
```

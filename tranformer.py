"""
Field Embedding Transformer
----------------------------
Learns a vector embedding for short text fields (names, addresses, etc.)
such that DIFFERENT SURFACE FORMS of the SAME entity map to SIMILAR vectors
(high cosine similarity), while different entities map to dissimilar vectors.

Approach: character-level Transformer encoder + mean pooling + L2 normalize,
trained with triplet loss on (anchor, augmented-positive, random-negative)
triples generated automatically from your corpus. No labeled pairs needed
to get started.

Usage:
    python field_embedding_transformer.py
"""

import math
import random
import string

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Character-level tokenizer
# ---------------------------------------------------------------------------
class CharTokenizer:
    """Maps raw strings to fixed-length index sequences at the character level.
    Character-level (not word/subword) is what gives robustness to typos,
    abbreviations, and punctuation/spacing differences common in names/addresses.
    """

    def __init__(self, max_len: int = 64):
        chars = string.ascii_lowercase + string.digits + " ,.-#/'&"
        self.vocab = {c: i + 2 for i, c in enumerate(chars)}
        self.vocab["<pad>"] = 0
        self.vocab["<unk>"] = 1
        self.max_len = max_len
        self.vocab_size = len(self.vocab) + 1  # headroom for unseen chars

    def encode(self, text: str):
        text = text.lower()[: self.max_len]
        ids = [self.vocab.get(c, 1) for c in text]
        ids += [0] * (self.max_len - len(ids))
        return ids

    def batch_encode(self, texts):
        return torch.tensor([self.encode(t) for t in texts], dtype=torch.long)


# ---------------------------------------------------------------------------
# 2. Positional encoding
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# 3. The encoder model
# ---------------------------------------------------------------------------
class FieldEmbeddingTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        embed_dim: int = 64,
        max_len: int = 64,
        pad_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encode = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.project = nn.Linear(d_model, embed_dim)

    def forward(self, x):
        pad_mask = x == self.pad_idx  # True where padded, shape (B, L)
        emb = self.token_embed(x)
        emb = self.pos_encode(emb)
        out = self.transformer(emb, src_key_padding_mask=pad_mask)

        # mean-pool over real (non-pad) tokens only
        keep = (~pad_mask).unsqueeze(-1).float()
        pooled = (out * keep).sum(1) / keep.sum(1).clamp(min=1e-6)

        embedding = self.project(pooled)
        embedding = F.normalize(embedding, p=2, dim=1)  # unit vectors -> cosine sim
        return embedding


# ---------------------------------------------------------------------------
# 4. Automatic augmentation -> synthetic positive pairs
# ---------------------------------------------------------------------------
ABBREV_MAP = {
    "street": "st", "road": "rd", "avenue": "ave", "apartment": "apt",
    "boulevard": "blvd", "drive": "dr", "lane": "ln", "court": "ct",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "building": "bldg", "floor": "fl", "limited": "ltd", "company": "co",
}


def augment(text: str) -> str:
    """Produces a plausible alternate way of writing the same field:
    abbreviation swaps, punctuation drops, a random typo, occasional
    word-order jitter. Used to synthesize positive pairs for training."""
    words = text.lower().split()
    out_words = []
    for w in words:
        core = w.strip(",.#")
        if random.random() < 0.35 and core in ABBREV_MAP:
            out_words.append(ABBREV_MAP[core])
        else:
            out_words.append(w)
    variant = " ".join(out_words)

    if random.random() < 0.3:
        variant = variant.replace(",", "")
    if random.random() < 0.2 and len(variant) > 4:
        pos = random.randint(0, len(variant) - 1)
        variant = variant[:pos] + random.choice(string.ascii_lowercase) + variant[pos + 1:]
    if random.random() < 0.15 and len(out_words) > 2:
        idx = list(range(len(out_words)))
        random.shuffle(idx)
        variant = " ".join(out_words[i] for i in idx)

    return variant


# ---------------------------------------------------------------------------
# 5. Training loop (triplet loss: anchor / augmented-positive / negative)
# ---------------------------------------------------------------------------
def train(model, tokenizer, corpus, epochs=30, batch_size=32, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    triplet_loss_fn = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1 - F.cosine_similarity(x, y), margin=0.3
    )

    for epoch in range(epochs):
        random.shuffle(corpus)
        total_loss, n_batches = 0.0, 0

        for i in range(0, len(corpus), batch_size):
            batch = corpus[i : i + batch_size]
            if len(batch) < 2:
                continue

            anchors = batch
            positives = [augment(t) for t in batch]
            # naive random negatives; swap in real "known different entity"
            # pairs if you have labels, for cleaner separation
            negatives = [random.choice(corpus) for _ in batch]

            a = model(tokenizer.batch_encode(anchors).to(device))
            p = model(tokenizer.batch_encode(positives).to(device))
            n = model(tokenizer.batch_encode(negatives).to(device))

            loss = triplet_loss_fn(a, p, n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg = total_loss / max(n_batches, 1)
            print(f"epoch {epoch + 1:3d}/{epochs}  loss={avg:.4f}")

    return model


# ---------------------------------------------------------------------------
# 6. Inference helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def get_embedding(model, tokenizer, text, device="cpu"):
    model.eval()
    ids = tokenizer.batch_encode([text]).to(device)
    return model(ids)[0].cpu()


def cosine_sim(vec_a, vec_b) -> float:
    return F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()

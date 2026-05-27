# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Dávid Navrátil <david.navratil2016@gmail.com>
"""
zeta/tokenizer.py  —  Byte and DNA Codon Tokenizers
====================================================
This module provides two native tokenizers:

    Byte tokenizer    : V = 256 tokens (0..255), each byte is a token
    DNA codon tokenizer : V = 64 tokens (AAA..TTT), 3-base codons

Token IDs are in Z_256 (vocabulary space).  The model embeds them into
Z_13[eta] via ZRingEmbed and predicts modulo 13 (ring characteristic).

These tokenizers are pure Python and are not in the forward data path.

Version : 7.0.0 (final)
Author  : Dávid Navrátil
License : CC-BY-NC-4.0
"""

from __future__ import annotations
from typing import List, Dict
import torch

from .constants import ZetaConfig, P

# Vocabulary size (token IDs 0..255 for byte tokenizer)
# This is NOT the ring characteristic P=13.
# Token IDs are embedded via ZRingEmbed into Z_13[η] as η^{(v·D+d)}.
# Model predictions are mod 13 (ring char), targets are mod 13.
VOCAB = ZetaConfig.V  # 256


def tokenise(text: str) -> List[int]:
    return [b % VOCAB for b in text.encode('utf-8', errors='replace')]


def detokenise(ids: List[int]) -> str:
    return bytes(i % 256 for i in ids).decode('utf-8', errors='replace')


def pad_batch(seqs: List[List[int]], pad: int = 0,
              maxlen: int = ZetaConfig.CTX) -> torch.Tensor:
    B = len(seqs)
    out = torch.full((B, maxlen), pad, dtype=torch.long)
    for b, s in enumerate(seqs):
        L = min(len(s), maxlen)
        out[b, :L] = torch.tensor(s[:L], dtype=torch.long)
    return out


# DNA codon tokenizer — 64 codons → 64-token vocab
_B = 'ACGT'
_DNA: Dict[str, int] = {
    a+b+c: i*16 + j*4 + k
    for i, a in enumerate(_B)
    for j, b in enumerate(_B)
    for k, c in enumerate(_B)
}


def dna_tokenise(seq: str) -> List[int]:
    seq = seq.upper()
    return [_DNA.get(seq[i:i+3], 0) for i in range(0, len(seq)-2, 3)]


# --- Hierarchical Trigram Tokenizer (general text, not just DNA) ---
# Compresses sequence by 3×: 3 bytes → 1 token via deterministic Z_13[eta] hash.

def trigram_tokenise(text: str) -> List[int]:
    """Group 3 bytes into 1 token.  Hash: (b1 + 7*b2 + 10*b3) % 256.
    Deterministic, no learned params.  Sequence length = ceil(len(bytes)/3)."""
    raw = text.encode('utf-8', errors='replace')
    out = []
    for i in range(0, len(raw) - 2, 3):
        b1, b2, b3 = raw[i], raw[i+1], raw[i+2]
        # CRT-style hash in Z_13[eta] spirit: coefficients (1, 7, 10)
        h = (b1 + 7 * b2 + 10 * b3) % 256
        out.append(h)
    # Pad last incomplete group with zeros
    rem = len(raw) % 3
    if rem == 1:
        out.append(raw[-1] % 256)
    elif rem == 2:
        h = (raw[-2] + 7 * raw[-1]) % 256
        out.append(h)
    return out


def detrigram_tokenise(ids: List[int]) -> str:
    """Inverse is non-unique (lossy).  Reconstructs approximate bytes."""
    raw = bytearray()
    for h in ids:
        # Approximate inverse: distribute h into 3 bytes proportionally
        b1 = h % 13
        b2 = (h // 13) % 13
        b3 = (h // 169) % 13
        raw.extend([b1, b2, b3])
    return bytes(raw).decode('utf-8', errors='replace')


def pad_batch_trigram(seqs: List[List[int]], pad: int = 0,
                      maxlen: int = ZetaConfig.CTX) -> torch.Tensor:
    """Pad trigram sequences to (B, maxlen)."""
    B = len(seqs)
    out = torch.full((B, maxlen), pad, dtype=torch.long)
    for b, s in enumerate(seqs):
        L = min(len(s), maxlen)
        out[b, :L] = torch.tensor(s[:L], dtype=torch.long)
    return out

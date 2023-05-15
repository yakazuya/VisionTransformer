#! /usr/bin/env python3

import torch
import torch.nn as nn
from mh_self_Attention import MultiHeadSelfAttention

class EncoderBlock(nn.Module):
    def __init__(self,
                 emb_dim    : int = 384,
                 head       : int = 8,
                 hidden_dim : int = 384 * 4,
                 dropout    : float = 0.
                 ):
        """
        emb_dim     : 埋め込み後のベクトルの長さ
        head        : ヘッドの数
        hidden_dim  : Encoder Blockにおける中間層の長さ(paper default:4倍)
        dropout     : ドロップアウト率
        """

        super(EncoderBlock, self).__init__()
        
        # 1つめのLayer Normalization
        self.ln1 = nn.LayerNorm(emb_dim)

        # MHSA
        self.MHSA = MultiHeadSelfAttention(
            emb_dim = emb_dim,
            head = head,
            dropout = dropout
        )

        # 2つめのLayer Normalization
        self.ln2 = nn.LayerNorm(emb_dim)

        # MLP
        self.MLP = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, z : torch.Tensor) -> torch.Tensor:
        """
        引数    : Encoder Blockへの入力(B, N, D)
        返り値  : Encoder Blockの出力(B, N, D)
        """

        # 前半部(1)
        out = self.MHSA(self.ln1(z)) + z
        # 後半部(2)
        out = self.MLP(self.ln2(z)) + out

        return out
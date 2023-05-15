#! /usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 emb_dim    : int = 384,
                 head       : int = 3,
                 dropout    : float=0.
                 ):
        """
        emb_dim     : 埋め込み後のベクトルの長さ
        head        : ヘッドの数
        dropout     : ドロップアウト率
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // head
        self.sqrt_dh = self.head_dim ** 0.5

        # 入力をq・k・vに埋め込むための線形層
        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)

        # 実装時にはドロップアウトも行う
        self.attn_drop = nn.Dropout(dropout)

        # MHSAの結果を出力に埋め込むための線形層
        ## 入力サイズが384である1つの全結合層と、Dropout率が0.2のDropout層を結合
        """
        nn.Sequentialは、順序に沿って適用する一連のレイヤーを定義するために使用される。
        これにより、より複雑なネットワークをより簡単に構築できる
        """
        self.w_o = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        引数    : MHSAへの入力(B, N(トークン数), D(埋め込みベクトルの長さ))
        返り値  : MHSAの出力(B, N(トークン数), D(埋め込みベクトルの長さ))
        """

        batch_size, num_patch, _ = z.size()

        # 埋め込み (B, N, D) -> (B, N, D)
        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)

        # 各ヘッドに分割 (B, N, D) -> (B, N, h, D//h)
        q = q.view(batch_size, num_patch, self.head, self.head_dim)
        k = k.view(batch_size, num_patch, self.head, self.head_dim)
        v = v.view(batch_size, num_patch, self.head, self.head_dim)

        # Self-Attention用に順番を変更(B, N, h, D//h) -> (B, h, N, D//h)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)


        # 内積
        k_T = k.transpose(2, 3)     # (B, h, N, D//h) -> (B, h, D//h, N)
        ## (B, h, N, D//h) x (B, h, D//h, N) -> (B, h, N, N)
        dots = (q @ k_T) / self.sqrt_dh
        # Softmax
        attn = F.softmax(dots, dim = 1)
        attn = self.attn_drop(attn)

        # 加重和　(B, h, N, N) x (B, h, N, D//h) -> (B, h, N, D//h)
        out = attn @ v

        # 元に戻す
        out = out.transpose(1, 2)   # (B, h, N, D//h) -> (B, N, h, D//h)
        out = out.reshape(batch_size, num_patch, self.emb_dim)  # B, N, h, D//h) -> (B, N, D)

        # 出力層
        out = self.w_o(out)
        return out
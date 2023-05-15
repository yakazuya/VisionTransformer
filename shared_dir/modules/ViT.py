#! /usr/bin/env python3

import torch
import torch.nn as nn
from modules.Encoder_block import EncoderBlock
from modules.Input_Layer import InputLayer
from modules.mh_self_Attention import MultiHeadSelfAttention

class ViT(nn.Module):
    def __init__(self,
                 in_channels    : int = 3,
                 num_classes    : int = 10,
                 emb_dim        : int = 384,
                 num_patch_row  : int = 2,
                 image_size     : int = 32,
                 num_blocks     : int = 7,
                 head           : int = 8,
                 hidden_dim     : int = 384 * 4,
                 dropout        : float = 0.
                 ):
        """
        in_channels    : 入力画像のチャンネル数
        num_classes     : 画像分類のクラス数
        emb_dim         : 埋め込み後のベクトルの長さ
        num_patch_row   : 1辺のパッチの数
        image_size      : 入力画像の1辺の長さ(画像は正方形)
        num_blocks      : Encoderのblock数
        head            : ヘッドの数
        hidden_dim      : Encoder Blockにおける中間層の長さ(paper default:4倍)
        dropout         : ドロップアウト率
        """

        super().__init__()
        
        # Input Layer
        self.input_layer = InputLayer(
            in_channels,
            emb_dim,
            num_patch_row,
            image_size
        )

        # Encoder 
        self.encoder = nn.Sequential(*[EncoderBlock(emb_dim=emb_dim, head=head,hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_blocks)])

        # MLP
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        引数    : 入力画像(B, C, H, W)
        返り値  : ViTの出力(B, M(クラス数))
        """

        # Input Layer (B, C, H, W) -> (B, N, D)
        out = self.input_layer(x)
        
        # Encoder (B, N, D) -> (B, N, D)
        out = self.encoder(out)
        # クラストークンのみを引き抜く
        cls_token = out[:,0]

        pred = self.mlp_head(cls_token)

        return pred
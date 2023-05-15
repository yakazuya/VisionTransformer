#! /usr/bin/env python3

import torch
import torch.nn as nn

class InputLayer(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 emb_dim: int = 384,
                 num_patch_row: int = 2,
                 image_size: int = 32
                 ):
        """
        in_channels     : 入力画像のch数
        emb_dim         : 埋め込み後のベクトルの長さ
        num_patch_row  : 高さ方向のパッチの数
        image_size      : 入力画像のサイズ
        """

        """
        nn.moduleクラスのコンストラクタを呼び出す
        初期化しているイメージ
        """
        super(InputLayer, self).__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.num_patch_row = num_patch_row
        self.image_size = image_size

        # パッチの数(num_patch_row = 2の場合N_p=4)
        self.num_patch = self.num_patch_row ** 2

        # パッチの大きさ(image_size = 224, num_patch_row = 2の場合、高さ方向に112のサイズで分割)
        self.patch_size = int(self.image_size // self.num_patch_row)    # 切り捨て除算

        # 入力画像のパッチへの分割とパッチの埋め込みを同時に行う層
        self.patch_emb_layer = nn.Conv2d(
            in_channels = self.in_channels,
            out_channels = self.emb_dim,
            kernel_size = self.patch_size,
            stride = self.patch_size
        )

        # クラストークン
        """
        nn.Parameter    : 学習可能なテンソル
        学習可能な適当な変数を作るイメージ
        例：畳み込みニューラルネットワークのフィルターやバイアス、全結合層の重みやバイアスなどがパラメーターとして扱われ、学習によって最適化される
        """
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, emb_dim)
        )

        # 位置埋め込み
        self.pos_emb = nn.Parameter(
            torch.randn(1, self.num_patch +1, emb_dim)
        )

    # torch.Tensor型の入力指定・出力がtorch.Tensor
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        引数    : input image(B, C, H, W)
        返り値  : ViTへの入力(B, N(トークン数), D(埋め込みベクトルの長さ))
        """

        # パッチの埋め込みとflatten
        """
        (B, C, H, W) -> (B, D, H/P, W/P)
        P   : パッチの1辺の長さ
        """
        z_0 = self.patch_emb_layer(x)

        # パッチの平坦化(flatten)
        """
        (B, D, H/P, W/P) -> (B, D, Np)
        Np  : パッチ数(= H*W / P^2)         全体の面積 / パッチ1つあたりの面積
        """
        z_0 = z_0.flatten(start_dim = 2, end_dim = 3)        # 最後の2つの次元を平坦化
        # z_0 = z_0.flatten(2)

        # 軸の入れ替え (B, D, Np) -> (B, Np, D)
        z_0 = z_0.transpose(1,2)

        # クラストークンの結合
        """
        (B, Np, D) -> (B, N(=Np+1), D)
        ! クラストークンの形状 : (1, 1, D)
        -> repeatメソッドによって(B, 1, D)に変更してからパッチの埋め込みとの結合を行う
        repeatメソッド  : 配列を指定された回数繰り返す
        x.size(0)回(1行×1列)の行列として繰り返される
        """
        z_0 = torch.cat(
            [self.cls_token.repeat(repeats=(x.size(0), 1, 1)), z_0], dim = 1
        )

        # 位置埋め込みの加算
        """
        (B, N, D) -> (B, N, D)
        """
        z_0 = z_0 + self.pos_emb

        return z_0
# -*- coding: utf-8 -*-
"""章をまたいで使う損失関数。

基礎編「学習の仕組み」および実装編「不均衡への実装的対処」で扱ったものを、
そのまま学習ループに差し込める形にまとめた。医療画像は陽性画素が全体の1%に
満たないことも珍しくなく、素のCrossEntropyでは「全部陰性」に収束する。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """1 - Dice。重なりを直接最適化するので、陽性の少ないセグメンテーションに強い。

    logits: (N, 1, ...) または (N, ...)。target は同形の 0/1。
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits).flatten(1)
        t = target.float().flatten(1)
        inter = (p * t).sum(1)
        return (1.0 - (2.0 * inter + self.eps) / (p.sum(1) + t.sum(1) + self.eps)).mean()


class TverskyLoss(nn.Module):
    """Diceの一般化。beta を大きくすると偽陰性（見逃し）の罰が重くなる。

    alpha + beta = 1 で使うのが通例。alpha=beta=0.5 のとき Dice と一致する。
    **見逃しを減らしたい課題では beta を上げる** ― ただし偽陽性は必ず増える。
    どこに置くかは臨床側と一緒に決める判断であって、既定値で済ませない。
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6):
        super().__init__()
        self.alpha, self.beta, self.eps = alpha, beta, eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits).flatten(1)
        t = target.float().flatten(1)
        tp = (p * t).sum(1)
        fp = (p * (1 - t)).sum(1)
        fn = ((1 - p) * t).sum(1)
        return (1.0 - (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)).mean()


class FocalLoss(nn.Module):
    """易しい例の寄与を下げ、難しい例に学習を集中させる。

    gamma=0 で通常の binary cross entropy に戻る。alpha は陽性クラスの重み。
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        t = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        p_t = torch.exp(-bce)                       # 正解クラスの予測確率
        a_t = self.alpha * t + (1 - self.alpha) * (1 - t)
        return (a_t * (1 - p_t) ** self.gamma * bce).mean()


class DiceBCELoss(nn.Module):
    """Dice と BCE の和。実務でもっともよく使う組み合わせ。

    Dice だけだと学習初期に勾配が立ちにくく、BCE だけだと不均衡に負ける。
    """

    def __init__(self, w_dice: float = 1.0, w_bce: float = 1.0):
        super().__init__()
        self.dice = DiceLoss()
        self.w_dice, self.w_bce = w_dice, w_bce

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target.float())
        return self.w_dice * self.dice(logits, target) + self.w_bce * bce

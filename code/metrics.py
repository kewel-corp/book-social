# -*- coding: utf-8 -*-
"""章をまたいで使う評価指標。

本文（基礎編の評価の章、社会実装編の評価・検証と説明可能性の章）で扱った指標を、
テストの書ける形にまとめたもの。**壊れても例外を投げず、もっともらしい数字を返す**
たぐいの関数なので、tests/ の単体テストとセットで使うこと。
"""
from __future__ import annotations

import numpy as np


def confusion(y_true, y_pred):
    """二値の混同行列を (TP, FP, FN, TN) で返す。入力は 0/1 の配列。"""
    y_true = np.asarray(y_true).astype(bool).ravel()
    y_pred = np.asarray(y_pred).astype(bool).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true と y_pred の形が違う: %s vs %s" % (y_true.shape, y_pred.shape))
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    return tp, fp, fn, tn


def sensitivity(y_true, y_pred):
    """感度（再現率）＝ TP / (TP + FN)。分母は「実際に陽性だった数」。"""
    tp, _fp, fn, _tn = confusion(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) else float("nan")


def specificity(y_true, y_pred):
    """特異度＝ TN / (TN + FP)。分母は「実際に陰性だった数」。"""
    _tp, fp, _fn, tn = confusion(y_true, y_pred)
    return tn / (tn + fp) if (tn + fp) else float("nan")


def precision(y_true, y_pred):
    """適合率（PPV）＝ TP / (TP + FP)。分母は「陽性と判定した数」。

    感度と適合率は分母が違う。ここを取り違えると、有病率の低い課題で
    「感度は高いのに現場では使えない」モデルを良いと誤認する。
    """
    tp, fp, _fn, _tn = confusion(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) else float("nan")


def dice(pred_mask, true_mask, eps: float = 1e-7):
    """Dice係数＝ 2|A∩B| / (|A|+|B|)。セグメンテーションの重なり具合。"""
    a = np.asarray(pred_mask).astype(bool)
    b = np.asarray(true_mask).astype(bool)
    if a.shape != b.shape:
        raise ValueError("マスクの形が違う: %s vs %s" % (a.shape, b.shape))
    inter = np.sum(a & b)
    return float((2.0 * inter + eps) / (a.sum() + b.sum() + eps))


def iou(pred_mask, true_mask, eps: float = 1e-7):
    """IoU（Jaccard）＝ |A∩B| / |A∪B|。Diceより厳しく出る。"""
    a = np.asarray(pred_mask).astype(bool)
    b = np.asarray(true_mask).astype(bool)
    if a.shape != b.shape:
        raise ValueError("マスクの形が違う: %s vs %s" % (a.shape, b.shape))
    inter = np.sum(a & b)
    union = np.sum(a | b)
    return float((inter + eps) / (union + eps))


def equiv_ellipse_minor_axis_mm(mask, spacing_mm=(1.0, 1.0)):
    """2次元マスクの「等価楕円の短軸長」（mm）を二次モーメントから求める。

    二次モーメントが同じ楕円の短軸長で、skimage の ``minor_axis_length`` と同じ定義。
    **輪郭上で実測する臨床の短径（RECIST などの計測規約）とは別物**で、形状によりずれる
    （長方形なら幅の約1.155倍）。基準に沿った計測を製品に載せるなら、別に実装して
    読影者の実測値と比べる（実装編「転移を数える ― リンパ節という難物」の節）。

    **係数を取り違えると値が半分／二倍になり、しかも例外は出ない。** 寸法の分かっている
    楕円マスクで必ずテストすること（社会実装編「臨床ロジックは、単体テストで守る」の節）。

    spacing_mm は (行方向, 列方向) の画素間隔。空・1〜2画素のマスクは評価不能として None を返す
    （0.0 を返すと「短径 0 mm の節」という誤った数字が下流へ流れる）。
    """
    m = np.asarray(mask).astype(bool)
    pts = np.argwhere(m).astype(float)
    if len(pts) < 3:
        return None
    pts *= np.asarray(spacing_mm, dtype=float)   # 画素座標 → mm
    pts -= pts.mean(axis=0)
    s = np.linalg.svd(pts, compute_uv=False)
    # 楕円の半径は s/sqrt(N)、径はその2倍。短軸は小さいほうの主軸。
    return float(4.0 * s[-1] / np.sqrt(len(pts)))


def short_axis_mm(mask, spacing_mm=(1.0, 1.0)):
    """後方互換の別名。名前が臨床の「短径」と紛らわしいため、新しいコードでは
    ``equiv_ellipse_minor_axis_mm`` を使うこと。"""
    return equiv_ellipse_minor_axis_mm(mask, spacing_mm)

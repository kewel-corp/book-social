# -*- coding: utf-8 -*-
"""章をまたいで使うデータ周りの部品。

いちばん重要なのは `split_by_patient` である。**同じ患者の画像が学習と評価に
またがった瞬間、評価指標は嘘をつく。** しかも成績は上がるので、気づけない。
医療AIで最も多い事故がこれで、本文でも繰り返し扱っている。
"""
from __future__ import annotations

import csv
import os
from typing import Sequence

import numpy as np


def split_by_patient(patient_ids: Sequence, ratios=(0.7, 0.15, 0.15), seed: int = 0):
    """**患者単位**で train/val/test の添字を分ける。

    画像単位でシャッフルしてはいけない。同一患者の別スライス・別方向が
    学習側と評価側に散ると、モデルは「その患者を覚えている」だけで高得点を出す。

    戻り値: (train_idx, val_idx, test_idx) ― それぞれ元の配列に対する添字の配列。
    """
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios の合計が1になっていない: %s" % (ratios,))
    ids = np.asarray(patient_ids)
    uniq = np.unique(ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(round(n * ratios[0]))
    n_va = int(round(n * ratios[1]))
    groups = (uniq[:n_tr], uniq[n_tr:n_tr + n_va], uniq[n_tr + n_va:])
    return tuple(np.where(np.isin(ids, g))[0] for g in groups)


def assert_no_patient_leak(*index_sets, patient_ids):
    """分割にまたがった患者がいないことを確かめる。学習を始める前に必ず呼ぶ。"""
    ids = np.asarray(patient_ids)
    sets = [set(ids[np.asarray(ix)]) for ix in index_sets]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            dup = sets[i] & sets[j]
            if dup:
                raise AssertionError(
                    "患者IDが分割をまたいでいる（%d名）: %s" % (len(dup), sorted(dup)[:5]))
    return True


def read_dataset_csv(path: str):
    """`dataset.csv`（image_path, label, patient_id, ...）を読む。

    列が足りない・パスが存在しない場合は、学習が始まる前に落とす。
    「学習は回ったが、実は半分のファイルが無かった」を防ぐため。
    """
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        raise ValueError("空の dataset.csv: %s" % path)
    need = {"image_path", "label", "patient_id"}
    missing = need - set(rows[0].keys())
    if missing:
        raise ValueError("dataset.csv に必要な列がない: %s" % sorted(missing))
    base = os.path.dirname(os.path.abspath(path))
    ng = [r["image_path"] for r in rows
          if not os.path.exists(os.path.join(base, r["image_path"]))]
    if ng:
        raise FileNotFoundError("画像が見つからない（%d件）: %s" % (len(ng), ng[:3]))
    return rows


def resize_pair(image, mask, size, order_image: int = 1):
    """画像とマスクを同じ大きさに揃える。**補間の種類を取り違えないこと。**

    画像（HU値・輝度）には線形などの滑らかな補間を使ってよいが、
    **ラベル（マスク）には必ず最近傍補間**を使う。線形補間をかけると、
    クラス0とクラス3のあいだに存在しない「1.5」が生まれ、静かに壊れる。
    """
    from skimage.transform import resize as _resize
    img = _resize(np.asarray(image), size, order=order_image,
                  preserve_range=True, anti_aliasing=order_image > 0)
    msk = _resize(np.asarray(mask), size, order=0,           # order=0 が最近傍
                  preserve_range=True, anti_aliasing=False)
    return img.astype(np.float32), np.rint(msk).astype(np.int64)

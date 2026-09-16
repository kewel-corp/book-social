# -*- coding: utf-8 -*-
"""臨床ロジックの単体テスト。

社会実装編「臨床ロジックは、単体テストで守る」の節の考え方を、そのまま形にしたもの。
ここにある関数はどれも、壊れても例外を投げない。もっともらしい数字を返し続ける。
だから「答えの分かる人工の入力」で確かめる以外に、守る方法がない。

    pytest tests/ -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import datasets as D  # noqa: E402
import metrics as M   # noqa: E402


# --- 混同行列と、そこから出る率 -------------------------------------------------

def test_confusion_and_rates():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    assert M.confusion(y_true, y_pred) == (2, 1, 1, 4)
    assert M.sensitivity(y_true, y_pred) == pytest.approx(2 / 3)
    assert M.specificity(y_true, y_pred) == pytest.approx(4 / 5)
    # 感度と適合率は分母が違う。ここを取り違えると有病率の低い課題で必ず誤る。
    assert M.precision(y_true, y_pred) == pytest.approx(2 / 3)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        M.confusion(np.zeros(4), np.zeros(5))


# --- Dice / IoU：完全一致・半分一致・不一致で手計算と合うこと --------------------

def test_dice_iou_known_values():
    a = np.zeros((10, 10), bool); a[:5, :] = True     # 50画素
    b = np.zeros((10, 10), bool); b[2:7, :] = True    # 50画素、重なり30画素
    assert M.dice(a, a) == pytest.approx(1.0, abs=1e-5)
    assert M.iou(a, a) == pytest.approx(1.0, abs=1e-5)
    assert M.dice(a, b) == pytest.approx(2 * 30 / (50 + 50), abs=1e-4)
    assert M.iou(a, b) == pytest.approx(30 / 70, abs=1e-4)
    c = np.zeros((10, 10), bool); c[8:, :] = True     # 重なりゼロ
    assert M.dice(a, c) == pytest.approx(0.0, abs=1e-5)


# --- 病変の径：寸法の分かっている楕円で、実寸どおりの値が返ること ----------------

def test_short_axis_on_known_ellipse():
    """半径 20px × 10px の楕円 → 短径 20mm（spacing 1mm）。

    係数を 4.0 から 2.0 に間違えると半分になる。例外は出ない。だからここで止める。
    """
    yy, xx = np.mgrid[-40:41, -40:41]
    ellipse = ((xx / 20.0) ** 2 + (yy / 10.0) ** 2) <= 1.0
    assert M.short_axis_mm(ellipse, (1.0, 1.0)) == pytest.approx(20.0, abs=1.0)


def test_short_axis_scales_with_spacing():
    yy, xx = np.mgrid[-40:41, -40:41]
    ellipse = ((xx / 20.0) ** 2 + (yy / 10.0) ** 2) <= 1.0
    one = M.short_axis_mm(ellipse, (1.0, 1.0))
    two = M.short_axis_mm(ellipse, (2.0, 2.0))
    assert two == pytest.approx(2 * one, rel=1e-6)


# --- 症例単位分割：同じ患者が分割をまたがないこと --------------------------------

def test_split_is_by_patient_not_by_image():
    patient_ids = np.repeat(np.arange(50), 4)      # 50人 × 4枚ずつ
    tr, va, te = D.split_by_patient(patient_ids, seed=1)
    assert len(tr) + len(va) + len(te) == len(patient_ids)
    D.assert_no_patient_leak(tr, va, te, patient_ids=patient_ids)


def test_leak_is_detected():
    patient_ids = np.repeat(np.arange(10), 2)
    with pytest.raises(AssertionError):
        D.assert_no_patient_leak([0, 1], [1, 2], patient_ids=patient_ids)


def test_split_is_reproducible_with_seed():
    pid = np.repeat(np.arange(30), 3)
    a = D.split_by_patient(pid, seed=7)
    b = D.split_by_patient(pid, seed=7)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


# --- リサイズ：ラベルに存在しないクラスIDが生まれないこと ------------------------

def test_label_resize_uses_nearest_neighbour():
    """マスクに線形補間をかけると、0と3のあいだに 1.5 のような値が生まれる。

    それを学習に流すと、存在しないクラスを学ぶ。ここで気づけないと発見が遅れる。
    """
    image = np.random.rand(20, 20)
    mask = np.zeros((20, 20), int); mask[5:15, 5:15] = 3
    _img2, mask2 = D.resize_pair(image, mask, (40, 40))
    assert set(np.unique(mask2)).issubset({0, 3})


def test_minor_axis_is_equivalent_ellipse_not_contour():
    # 長方形（高さ10・幅40）では、等価楕円の短軸長は幅の約1.155倍になる（輪郭上の短径 10 ではない）
    m = np.zeros((50, 50), dtype=bool)
    m[10:20, 5:45] = True
    v = M.equiv_ellipse_minor_axis_mm(m, (1.0, 1.0))
    assert abs(v / 10.0 - 1.155) < 0.03


def test_minor_axis_empty_mask_is_not_measurable():
    assert M.equiv_ellipse_minor_axis_mm(np.zeros((5, 5), dtype=bool)) is None
    assert M.short_axis_mm(np.zeros((5, 5), dtype=bool)) is None   # 別名も同じ挙動

from pathlib import Path

import pytest

from pre_processing import split_train_val, split_train_val_test


def make_pairs(count):
    return [(Path(f"image_{index}.jpg"), Path(f"mask_{index}.png")) for index in range(count)]


def test_custom_split_sizes_are_reproducible():
    pairs = make_pairs(100)

    first_split = split_train_val_test(pairs, 0.7, 0.15, 0.15, seed=42)
    second_split = split_train_val_test(pairs, 0.7, 0.15, 0.15, seed=42)

    assert [len(split) for split in first_split] == [70, 15, 15]
    assert first_split == second_split


def test_official_train_validation_split_keeps_all_training_pairs():
    pairs = make_pairs(20)

    train_pairs, val_pairs = split_train_val(pairs, val_split=0.2, seed=7)

    assert len(train_pairs) == 16
    assert len(val_pairs) == 4
    assert sorted(train_pairs + val_pairs) == sorted(pairs)


def test_custom_split_rejects_invalid_ratios():
    with pytest.raises(ValueError):
        split_train_val_test(make_pairs(10), 0.7, 0.2, 0.2, seed=1)


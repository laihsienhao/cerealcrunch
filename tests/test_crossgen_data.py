from pathlib import Path

import pytest

from aigc_detect.crossgen_data import list_samples_by_source, split_by_source_stratified


def test_list_samples_by_source_groups_by_prefix(tmp_path):
    for name in ["star_gan__img0.jpg", "star_gan__img1.jpg", "big_gan__img0.jpg", "notes.txt"]:
        (tmp_path / name).write_bytes(b"fake image bytes")

    grouped = list_samples_by_source(tmp_path, label=1)

    assert set(grouped.keys()) == {"star_gan", "big_gan"}
    assert len(grouped["star_gan"]) == 2
    assert len(grouped["big_gan"]) == 1
    assert all(label == 1 for samples in grouped.values() for _, label in samples)


def _make_groups(n_groups: int, n_per_group: int, label: int) -> dict[str, list[tuple[Path, int]]]:
    return {
        f"source_{i}": [(Path(f"source_{i}/img{j}.jpg"), label) for j in range(n_per_group)]
        for i in range(n_groups)
    }


def test_split_by_source_stratified_hits_exact_totals_and_every_group():
    groups = _make_groups(n_groups=25, n_per_group=200, label=1)  # mirrors ArtiFact's fake side

    train, val, test = split_by_source_stratified(groups, n_train=4000, n_val=500, n_test=500)

    assert len(train) == 4000
    assert len(val) == 500
    assert len(test) == 500

    for split_name, split in (("train", train), ("val", val), ("test", test)):
        sources_present = {path.parent.name for path, _ in split}
        assert sources_present == set(groups.keys()), f"{split_name} is missing a source"


def test_split_by_source_stratified_is_disjoint_and_covers_all_samples():
    groups = _make_groups(n_groups=8, n_per_group=625, label=0)  # mirrors ArtiFact's real side

    train, val, test = split_by_source_stratified(groups, n_train=4000, n_val=500, n_test=500)

    all_input = {path for samples in groups.values() for path, _ in samples}
    train_set, val_set, test_set = set(p for p, _ in train), set(p for p, _ in val), set(p for p, _ in test)

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert train_set | val_set | test_set == all_input


def test_split_by_source_stratified_rejects_mismatched_totals():
    groups = _make_groups(n_groups=2, n_per_group=10, label=1)  # 20 total available
    with pytest.raises(ValueError):
        split_by_source_stratified(groups, n_train=10, n_val=5, n_test=4)  # sums to 19 != 20

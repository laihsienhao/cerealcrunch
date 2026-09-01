from aigc_detect.artifact_data import group_by_top_folder, select_members


def _synthetic_listing() -> list[str]:
    names = ["ArtiFact/", "ArtiFact/Fake/", "ArtiFact/Real/"]
    for generator, count in [("star_gan", 3), ("big_gan", 2)]:
        names.append(f"ArtiFact/Fake/{generator}/")
        names += [f"ArtiFact/Fake/{generator}/img{i:06d}.jpg" for i in range(count)]
    for source, count in [("celebahq", 4), ("lsun", 4)]:
        names.append(f"ArtiFact/Real/{source}/")
        names += [f"ArtiFact/Real/{source}/img{i:06d}.jpg" for i in range(count)]
    return names


def test_group_by_top_folder_separates_real_and_fake():
    fake_groups, real_groups = group_by_top_folder(_synthetic_listing())

    assert set(fake_groups.keys()) == {"star_gan", "big_gan"}
    assert len(fake_groups["star_gan"]) == 3
    assert len(fake_groups["big_gan"]) == 2
    assert set(real_groups.keys()) == {"celebahq", "lsun"}
    assert len(real_groups["celebahq"]) == 4


def test_group_by_top_folder_ignores_directory_entries_and_non_images():
    names = _synthetic_listing() + ["ArtiFact/Fake/star_gan/readme.txt"]
    fake_groups, _ = group_by_top_folder(names)
    assert len(fake_groups["star_gan"]) == 3  # readme.txt excluded, dir entries excluded


def test_select_members_takes_n_per_fake_generator():
    fake_groups, real_groups = group_by_top_folder(_synthetic_listing())
    selected = select_members(fake_groups, real_groups, n_per_fake_generator=2, n_total_real=4)

    fake_selected = [s for s in selected if s[1] == "FAKE"]
    assert len(fake_selected) == 4  # 2 generators x 2 each
    by_generator = {}
    for _, _, generator in fake_selected:
        by_generator[generator] = by_generator.get(generator, 0) + 1
    assert by_generator == {"star_gan": 2, "big_gan": 2}


def test_select_members_spreads_real_total_evenly_across_sources():
    fake_groups, real_groups = group_by_top_folder(_synthetic_listing())
    selected = select_members(fake_groups, real_groups, n_per_fake_generator=1, n_total_real=6)

    real_selected = [s for s in selected if s[1] == "REAL"]
    assert len(real_selected) == 6
    by_source = {}
    for _, _, source in real_selected:
        by_source[source] = by_source.get(source, 0) + 1
    # 6 spread across 2 sources -> 3 each
    assert by_source == {"celebahq": 3, "lsun": 3}

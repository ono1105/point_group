"""
pytest entry points for CI.

Reuses validate.py to check that every group in data/index.json passes
every check. Each (group, check) pair becomes a parameterized test, so
failures show up individually in the test report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make project root importable when tests run from anywhere.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loader import list_groups, load_group  # noqa: E402
from validate import ALL_CHECKS, check_index_consistency  # noqa: E402


def _all_pairs():
    """Yield (group_name, check_function) for every group and every check."""
    for entry in list_groups():
        for check in ALL_CHECKS:
            yield entry["schoenflies"], check


@pytest.mark.parametrize(
    "group_name,check",
    list(_all_pairs()),
    ids=lambda x: x.__name__ if callable(x) else x,
)
def test_group_check(group_name, check):
    g = load_group(group_name)
    result = check(g)
    assert result.ok, (
        f"{group_name} failed {result.name}:\n"
        + "\n".join(result.errors)
    )


def test_index_loadable():
    """index.json must list at least one group."""
    groups = list_groups()
    assert len(groups) >= 1, "data/index.json must register at least one group"


def test_all_groups_loadable():
    """Every entry in index.json must point to a readable file."""
    for entry in list_groups():
        g = load_group(entry["schoenflies"])
        assert g.schoenflies == entry["schoenflies"]
        assert g.order == entry["order"]


def test_index_consistency():
    """index.json と各 JSON ファイル間のグローバル整合性。

    群単位の ALL_CHECKS とは別に、index.json と各 JSON のメタデータが
    整合しているかをチェックする。
    """
    result = check_index_consistency()
    assert result.ok, (
        f"index_consistency failed:\n" + "\n".join(result.errors)
    )


def test_group_ids_are_sequential():
    """index.json の id は 1 から連番でなければならない。"""
    groups = list_groups()
    ids = [g["id"] for g in groups]
    expected = list(range(1, len(ids) + 1))
    assert ids == expected, (
        f"id が連番でない: {ids} （期待: {expected}）"
    )


def test_no_duplicate_schoenflies():
    """index.json 内で schoenflies 名が重複していないこと。"""
    names = [g["schoenflies"] for g in list_groups()]
    duplicates = [n for n in set(names) if names.count(n) > 1]
    assert not duplicates, f"重複している schoenflies 名: {duplicates}"

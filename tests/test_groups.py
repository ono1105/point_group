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
from validate import ALL_CHECKS              # noqa: E402


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

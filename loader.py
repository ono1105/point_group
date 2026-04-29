"""
Point group data loader.

JSON data files in data/ are loaded into PointGroup objects with convenience
methods for group operations.

Usage:
    from loader import load_group, load_all_groups, list_groups

    g = load_group("C3v")
    print(g.order)                  # 6
    print(g.multiply(1, 3))         # C3 * sigma_v

    for g in load_all_groups():
        print(g.schoenflies, g.order)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
INDEX_FILE = DATA_DIR / "index.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SymmetryElement:
    """A single symmetry operation."""
    index: int
    schoenflies: str
    seitz: str
    matrix: np.ndarray            # shape (3, 3), float
    type: str                     # 'identity', 'rotation', 'reflection', ...
    element_order: int            # smallest n with g^n = E
    inverse: int                  # index of inverse element

    # Optional geometry hints (may be None)
    axis: Optional[list] = None
    plane_normal: Optional[list] = None
    rotation_angle_deg: Optional[float] = None
    translation: list = field(default_factory=lambda: [0, 0, 0])


@dataclass
class Subgroup:
    """A subgroup of a point group."""
    schoenflies: str
    element_indices: list
    order: int
    index_in_parent: int
    is_normal: bool
    is_trivial: bool = False
    is_whole_group: bool = False
    is_maximal: bool = False
    coset_representatives: list = field(default_factory=list)
    quotient_group: Optional[str] = None
    note: str = ""


@dataclass
class PointGroup:
    """A crystallographic point group."""
    schoenflies: str
    hermann_mauguin: str
    crystal_system: str
    order: int
    abelian: bool
    generators: list
    elements: list                # list[SymmetryElement]
    mult_table: np.ndarray        # shape (n, n), int
    conjugacy_classes: list
    conjugacy_class_names: list
    center: list
    commutator_subgroup: list
    abelianization: str
    is_symmorphic: bool
    subgroups: list               # list[Subgroup]
    normal_subgroup_indices: list

    # Original raw dict (for debugging / forward-compat fields)
    _raw: dict = field(default_factory=dict, repr=False)

    # ------ basic group operations ------

    def multiply(self, i: int, j: int) -> int:
        """Return the index of g_i * g_j (g_j applied first)."""
        return int(self.mult_table[i, j])

    def inverse(self, i: int) -> int:
        """Return the index of the inverse of g_i."""
        return self.elements[i].inverse

    def conjugate(self, g: int, h: int) -> int:
        """Return the index of g h g^{-1}."""
        return self.multiply(self.multiply(g, h), self.inverse(g))

    def commutator(self, i: int, j: int) -> int:
        """Return the index of [g_i, g_j] = g_i g_j g_i^{-1} g_j^{-1}."""
        return self.multiply(
            self.multiply(self.multiply(i, j), self.inverse(i)),
            self.inverse(j),
        )

    # ------ name <-> index lookups ------

    def name_of(self, i: int) -> str:
        return self.elements[i].schoenflies

    def index_of(self, name: str) -> int:
        for e in self.elements:
            if e.schoenflies == name:
                return e.index
        raise KeyError(f"element '{name}' not found in group {self.schoenflies}")

    # ------ subgroup helpers ------

    def normal_subgroups(self) -> list:
        return [self.subgroups[i] for i in self.normal_subgroup_indices]

    def maximal_subgroups(self) -> list:
        return [s for s in self.subgroups if s.is_maximal]

    def find_subgroup(self, schoenflies: str) -> Subgroup:
        for s in self.subgroups:
            if s.schoenflies == schoenflies:
                return s
        raise KeyError(
            f"subgroup '{schoenflies}' not found in group {self.schoenflies}"
        )

    def __repr__(self) -> str:
        return (
            f"PointGroup({self.schoenflies}, HM={self.hermann_mauguin}, "
            f"order={self.order})"
        )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_index() -> dict:
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"index file not found: {INDEX_FILE}. "
            "Make sure data/index.json exists."
        )
    with open(INDEX_FILE, encoding="utf-8") as f:
        return json.load(f)


def list_groups() -> list:
    """Return the list of group metadata entries from index.json."""
    return _load_index()["groups"]


def _resolve_path(name_or_path: str | Path) -> Path:
    """Accept a Schoenflies name, a filename, or an absolute path."""
    p = Path(name_or_path)
    if p.is_absolute() and p.exists():
        return p
    if p.suffix == ".json" and (DATA_DIR / p).exists():
        return DATA_DIR / p

    # Treat as Schoenflies symbol; look up in index
    index = _load_index()
    for entry in index["groups"]:
        if entry["schoenflies"] == str(name_or_path):
            return DATA_DIR / entry["file"]
    raise KeyError(
        f"could not resolve '{name_or_path}'. "
        "Pass a Schoenflies name (e.g. 'C3v'), a filename in data/, "
        "or an absolute path."
    )


def load_group(name_or_path: str | Path) -> PointGroup:
    """Load a single point group by Schoenflies name or file path."""
    path = _resolve_path(name_or_path)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return _build_point_group(raw)


def load_all_groups() -> Iterator[PointGroup]:
    """Yield every group registered in index.json."""
    for entry in list_groups():
        yield load_group(entry["schoenflies"])


def _build_point_group(raw: dict) -> PointGroup:
    """Convert a raw dict from JSON into a PointGroup object."""
    elements = [
        SymmetryElement(
            index=e["index"],
            schoenflies=e["schoenflies"],
            seitz=e["seitz"],
            matrix=np.array(e["matrix"], dtype=float),
            type=e["type"],
            element_order=e["element_order"],
            inverse=e["inverse"],
            axis=e.get("axis"),
            plane_normal=e.get("plane_normal"),
            rotation_angle_deg=e.get("rotation_angle_deg"),
            translation=e.get("translation", [0, 0, 0]),
        )
        for e in raw["elements"]
    ]

    subgroups = [
        Subgroup(
            schoenflies=s["schoenflies"],
            element_indices=s["element_indices"],
            order=s["order"],
            index_in_parent=s["index_in_parent"],
            is_normal=s["is_normal"],
            is_trivial=s.get("is_trivial", False),
            is_whole_group=s.get("is_whole_group", False),
            is_maximal=s.get("is_maximal", False),
            coset_representatives=s.get("coset_representatives", []),
            quotient_group=s.get("quotient_group"),
            note=s.get("note", ""),
        )
        for s in raw["subgroups"]
    ]

    return PointGroup(
        schoenflies=raw["schoenflies"],
        hermann_mauguin=raw["hermann_mauguin"],
        crystal_system=raw["crystal_system"],
        order=raw["order"],
        abelian=raw["abelian"],
        generators=raw["generators"],
        elements=elements,
        mult_table=np.array(raw["multiplication_table"], dtype=int),
        conjugacy_classes=raw["conjugacy_classes"],
        conjugacy_class_names=raw.get("conjugacy_class_names", []),
        center=raw.get("center", []),
        commutator_subgroup=raw.get("commutator_subgroup", []),
        abelianization=raw.get("abelianization", ""),
        is_symmorphic=raw.get("is_symmorphic", True),
        subgroups=subgroups,
        normal_subgroup_indices=raw.get("normal_subgroups", []),
        _raw=raw,
    )


# ---------------------------------------------------------------------------
# Quick check when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"data dir: {DATA_DIR}")
    print(f"registered groups: {len(list_groups())}")
    for g in load_all_groups():
        print(f"  {g}")

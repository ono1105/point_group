"""
Point group data validator.

Runs all validation checks against every group registered in data/index.json.
On error, prints all violations and exits with status 1 (suitable for CI).

Usage:
    python validate.py                  # validate all groups
    python validate.py C3v Oh           # validate specific groups
    python validate.py --quiet          # only print failures
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable

import numpy as np

from loader import PointGroup, list_groups, load_group


# ---------------------------------------------------------------------------
# Error collection
# ---------------------------------------------------------------------------

class CheckResult:
    """Collects errors from a single check function."""
    def __init__(self, name: str):
        self.name = name
        self.errors: list[str] = []

    def fail(self, msg: str) -> None:
        self.errors.append(msg)

    @property
    def ok(self) -> bool:
        return not self.errors


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_identity(g: PointGroup) -> CheckResult:
    """Element 0 must act as the identity in the multiplication table."""
    r = CheckResult("identity")
    for i in range(g.order):
        if g.multiply(0, i) != i:
            r.fail(f"E * g_{i} = g_{g.multiply(0, i)} (expected g_{i})")
        if g.multiply(i, 0) != i:
            r.fail(f"g_{i} * E = g_{g.multiply(i, 0)} (expected g_{i})")
    return r


def check_associativity(g: PointGroup) -> CheckResult:
    """(a*b)*c == a*(b*c) for all triples."""
    r = CheckResult("associativity")
    n = g.order
    for i in range(n):
        for j in range(n):
            for k in range(n):
                left = g.multiply(g.multiply(i, j), k)
                right = g.multiply(i, g.multiply(j, k))
                if left != right:
                    r.fail(
                        f"({i}*{j})*{k}={left}, {i}*({j}*{k})={right}"
                    )
                    return r  # one failure is enough
    return r


def check_inverse(g: PointGroup) -> CheckResult:
    """Each element's declared inverse must satisfy g * g^-1 = E (both sides)."""
    r = CheckResult("inverse")
    for e in g.elements:
        if g.multiply(e.index, e.inverse) != 0:
            r.fail(f"g_{e.index} * g_{e.inverse} != E")
        if g.multiply(e.inverse, e.index) != 0:
            r.fail(f"g_{e.inverse} * g_{e.index} != E")
    return r


def check_closure(g: PointGroup) -> CheckResult:
    """Every entry of the multiplication table is a valid index."""
    r = CheckResult("closure")
    n = g.order
    for i in range(n):
        for j in range(n):
            k = int(g.mult_table[i, j])
            if not 0 <= k < n:
                r.fail(f"table[{i}][{j}] = {k} out of range")
    return r


def check_matrix_consistency(g: PointGroup, tol: float = 1e-9) -> CheckResult:
    """The product M_i @ M_j must equal M_table[i][j]."""
    r = CheckResult("matrix_consistency")
    n = g.order
    for i in range(n):
        for j in range(n):
            prod = g.elements[i].matrix @ g.elements[j].matrix
            k = g.multiply(i, j)
            if not np.allclose(prod, g.elements[k].matrix, atol=tol):
                r.fail(
                    f"M_{i} @ M_{j} != M_{k}\n"
                    f"  computed:\n{prod}\n"
                    f"  expected (M_{k}):\n{g.elements[k].matrix}"
                )
                return r
    return r


def check_element_order(g: PointGroup) -> CheckResult:
    """Declared element_order n must satisfy g^n = E and be the smallest such n."""
    r = CheckResult("element_order")
    for e in g.elements:
        # compute g^n iteratively
        cur = 0  # E
        for power in range(1, g.order + 1):
            cur = g.multiply(cur, e.index)
            if cur == 0:
                if power != e.element_order:
                    r.fail(
                        f"g_{e.index} ({e.schoenflies}): declared order "
                        f"{e.element_order}, actual {power}"
                    )
                break
        else:
            r.fail(f"g_{e.index} never returns to E within order bound")
    return r


def check_subgroup_closure(g: PointGroup) -> CheckResult:
    """Each declared subgroup is closed under multiplication."""
    r = CheckResult("subgroup_closure")
    for sub in g.subgroups:
        H = set(sub.element_indices)
        for i in H:
            for j in H:
                if g.multiply(i, j) not in H:
                    r.fail(
                        f"{sub.schoenflies}: {i}*{j} = {g.multiply(i, j)} "
                        f"escapes subgroup"
                    )
    return r


def check_subgroup_identity(g: PointGroup) -> CheckResult:
    """Every subgroup must contain the identity (index 0)."""
    r = CheckResult("subgroup_identity")
    for sub in g.subgroups:
        if 0 not in sub.element_indices:
            r.fail(f"{sub.schoenflies} does not contain E")
    return r


def check_subgroup_order(g: PointGroup) -> CheckResult:
    """Declared order matches len(element_indices)."""
    r = CheckResult("subgroup_order")
    for sub in g.subgroups:
        if sub.order != len(sub.element_indices):
            r.fail(
                f"{sub.schoenflies}: declared order {sub.order}, "
                f"actual {len(sub.element_indices)}"
            )
        if sub.index_in_parent != g.order // sub.order:
            r.fail(
                f"{sub.schoenflies}: declared index {sub.index_in_parent}, "
                f"actual {g.order // sub.order}"
            )
    return r


def check_lagrange(g: PointGroup) -> CheckResult:
    """Subgroup order divides parent order."""
    r = CheckResult("lagrange")
    for sub in g.subgroups:
        if g.order % sub.order != 0:
            r.fail(
                f"{sub.schoenflies}: |H|={sub.order} does not divide "
                f"|G|={g.order}"
            )
    return r


def check_normality_flags(g: PointGroup) -> CheckResult:
    """is_normal flags must agree with the actual conjugation test."""
    r = CheckResult("normality_flags")
    declared_normal = set()
    for k, sub in enumerate(g.subgroups):
        H = set(sub.element_indices)
        actual = all({g.conjugate(x, h) for h in H} == H for x in range(g.order))
        if actual != sub.is_normal:
            r.fail(
                f"{sub.schoenflies}: declared is_normal={sub.is_normal}, "
                f"actual={actual}"
            )
        if sub.is_normal:
            declared_normal.add(k)
    declared_in_list = set(g.normal_subgroup_indices)
    if declared_normal != declared_in_list:
        r.fail(
            f"normal_subgroups list {sorted(declared_in_list)} "
            f"disagrees with per-subgroup flags {sorted(declared_normal)}"
        )
    return r


def check_conjugacy_classes(g: PointGroup) -> CheckResult:
    """Declared conjugacy classes match actual conjugation orbits."""
    r = CheckResult("conjugacy_classes")
    declared = [set(c) for c in g.conjugacy_classes]

    # Compute actual classes
    seen = set()
    actual = []
    for h in range(g.order):
        if h in seen:
            continue
        orbit = {g.conjugate(x, h) for x in range(g.order)}
        actual.append(orbit)
        seen |= orbit

    # Same partition?
    if {frozenset(c) for c in declared} != {frozenset(c) for c in actual}:
        r.fail(
            f"declared classes {[sorted(c) for c in declared]} "
            f"differ from actual {[sorted(c) for c in actual]}"
        )
    return r


def check_abelian_flag(g: PointGroup) -> CheckResult:
    """The abelian flag must agree with the actual commutativity check."""
    r = CheckResult("abelian_flag")
    actual = all(
        g.multiply(i, j) == g.multiply(j, i)
        for i in range(g.order)
        for j in range(g.order)
    )
    if actual != g.abelian:
        r.fail(f"declared abelian={g.abelian}, actual={actual}")
    return r


def check_center(g: PointGroup) -> CheckResult:
    """Declared center matches Z(G) = {z : zg = gz for all g}."""
    r = CheckResult("center")
    actual = [
        i for i in range(g.order)
        if all(g.multiply(i, j) == g.multiply(j, i) for j in range(g.order))
    ]
    if set(actual) != set(g.center):
        r.fail(f"declared center={g.center}, actual={actual}")
    return r


def check_commutator_subgroup(g: PointGroup) -> CheckResult:
    """Declared commutator subgroup matches the actual derived subgroup."""
    r = CheckResult("commutator_subgroup")
    base = {
        g.commutator(i, j)
        for i in range(g.order)
        for j in range(g.order)
    }
    # closure under multiplication
    closure = set(base)
    changed = True
    while changed:
        changed = False
        new = set()
        for a in closure:
            for b in closure:
                p = g.multiply(a, b)
                if p not in closure:
                    new.add(p)
        if new:
            closure |= new
            changed = True
    if closure != set(g.commutator_subgroup):
        r.fail(
            f"declared {sorted(g.commutator_subgroup)}, "
            f"actual {sorted(closure)}"
        )
    return r


def check_metadata(g: PointGroup) -> CheckResult:
    """Sanity: order matches len(elements); generators are valid indices; etc."""
    r = CheckResult("metadata")
    if g.order != len(g.elements):
        r.fail(f"order={g.order} but len(elements)={len(g.elements)}")
    for idx, e in enumerate(g.elements):
        if e.index != idx:
            r.fail(f"elements[{idx}].index = {e.index}")
    for i in g.generators:
        if not 0 <= i < g.order:
            r.fail(f"generator index {i} out of range")
    if g.mult_table.shape != (g.order, g.order):
        r.fail(f"mult_table shape {g.mult_table.shape} != ({g.order}, {g.order})")
    return r


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

ALL_CHECKS: list[Callable[[PointGroup], CheckResult]] = [
    check_metadata,
    check_identity,
    check_closure,
    check_associativity,
    check_inverse,
    check_element_order,
    check_matrix_consistency,
    check_subgroup_identity,
    check_subgroup_order,
    check_subgroup_closure,
    check_lagrange,
    check_normality_flags,
    check_conjugacy_classes,
    check_abelian_flag,
    check_center,
    check_commutator_subgroup,
]


def validate_group(g: PointGroup, quiet: bool = False) -> list[CheckResult]:
    """Run every check on a single group and return the results."""
    if not quiet:
        print(f"=== {g.schoenflies} (order={g.order}) ===")
    results = []
    for check in ALL_CHECKS:
        res = check(g)
        results.append(res)
        if not quiet:
            mark = "OK" if res.ok else "NG"
            print(f"  [{mark}] {res.name}")
        if not res.ok and quiet:
            print(f"[NG] {g.schoenflies}: {res.name}")
        if not res.ok:
            for err in res.errors[:5]:
                indent = "      " if not quiet else "    "
                for line in err.splitlines():
                    print(f"{indent}{line}")
            if len(res.errors) > 5:
                print(f"      ... and {len(res.errors) - 5} more errors")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate point group data")
    parser.add_argument(
        "groups", nargs="*",
        help="Schoenflies names to validate (default: all in index.json)"
    )
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Only print failures")
    args = parser.parse_args()

    if args.groups:
        targets = args.groups
    else:
        targets = [entry["schoenflies"] for entry in list_groups()]

    if not targets:
        print("No groups registered in data/index.json")
        return 0

    total_failures = 0
    for name in targets:
        g = load_group(name)
        results = validate_group(g, quiet=args.quiet)
        total_failures += sum(1 for r in results if not r.ok)
        if not args.quiet:
            print()

    if total_failures:
        print(f"=== FAILED: {total_failures} check(s) failed ===")
        return 1
    print(f"=== PASSED: all {len(targets)} group(s) valid ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

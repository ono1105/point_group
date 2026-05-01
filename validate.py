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


def check_inverse_symmetry(g: PointGroup) -> CheckResult:
    """If g_i.inverse = j then g_j.inverse must be i."""
    r = CheckResult("inverse_symmetry")
    for e in g.elements:
        j = e.inverse
        if g.elements[j].inverse != e.index:
            r.fail(
                f"g_{e.index}.inverse = {j} but g_{j}.inverse = "
                f"{g.elements[j].inverse}"
            )
    return r


def check_conjugacy_partition(g: PointGroup) -> CheckResult:
    """Conjugacy classes must partition {0, ..., n-1} (cover and disjoint)."""
    r = CheckResult("conjugacy_partition")
    seen = set()
    total = 0
    for cc in g.conjugacy_classes:
        cc_set = set(cc)
        overlap = seen & cc_set
        if overlap:
            r.fail(f"共役類が重複: {sorted(overlap)} が複数の類に属する")
        seen |= cc_set
        total += len(cc)
    expected = set(range(g.order))
    if seen != expected:
        missing = expected - seen
        extra = seen - expected
        if missing:
            r.fail(f"共役類が要素を覆わない: 欠けている要素 {sorted(missing)}")
        if extra:
            r.fail(f"共役類に範囲外の要素: {sorted(extra)}")
    if total != g.order:
        r.fail(f"共役類サイズ合計 {total} ≠ order {g.order}")
    if g.conjugacy_class_names and len(g.conjugacy_class_names) != len(g.conjugacy_classes):
        r.fail(
            f"conjugacy_class_names 長 {len(g.conjugacy_class_names)} "
            f"≠ conjugacy_classes 長 {len(g.conjugacy_classes)}"
        )
    return r


def check_abelian_consequences(g: PointGroup) -> CheckResult:
    """Abelian groups: center = G, all conjugacy classes have size 1,
    and commutator subgroup = {E}."""
    r = CheckResult("abelian_consequences")
    if g.abelian:
        if len(g.center) != g.order:
            r.fail(
                f"abelian=True だが |center|={len(g.center)} "
                f"≠ |G|={g.order}（中心は全要素であるべき）"
            )
        for cc in g.conjugacy_classes:
            if len(cc) != 1:
                r.fail(f"abelian=True だが共役類 {cc} のサイズが 1 でない")
                break
        if set(g.commutator_subgroup) != {0}:
            r.fail(
                f"abelian=True だが commutator_subgroup="
                f"{sorted(g.commutator_subgroup)} ≠ {{0}}"
            )
    else:
        if set(g.commutator_subgroup) == {0}:
            r.fail(
                "abelian=False だが commutator_subgroup={0}"
                "（非可換なら交換子部分群は自明群より大きい）"
            )
    return r


def check_subgroup_flags(g: PointGroup) -> CheckResult:
    """is_trivial / is_whole_group flag consistency.

    - is_trivial=True ⟺ order = 1
    - is_whole_group=True ⟺ order = |G|
    - 自明群と全群は必ず正規
    - 自明群と全群はそれぞれ subgroups リストに 1 個ずつ
    """
    r = CheckResult("subgroup_flags")
    n_triv = 0
    n_whole = 0
    for sub in g.subgroups:
        if sub.is_trivial and sub.order != 1:
            r.fail(
                f"{sub.schoenflies}: is_trivial=True なのに order={sub.order}"
            )
        if sub.order == 1 and not sub.is_trivial:
            r.fail(f"{sub.schoenflies}: order=1 なのに is_trivial=False")
        if sub.is_whole_group and sub.order != g.order:
            r.fail(
                f"{sub.schoenflies}: is_whole_group=True なのに "
                f"order={sub.order} ≠ |G|={g.order}"
            )
        if sub.order == g.order and not sub.is_whole_group:
            r.fail(
                f"{sub.schoenflies}: order=|G| なのに is_whole_group=False"
            )
        if sub.is_trivial and not sub.is_normal:
            r.fail(f"{sub.schoenflies}: 自明群が is_normal=False（自明群は常に正規）")
        if sub.is_whole_group and not sub.is_normal:
            r.fail(f"{sub.schoenflies}: 全群が is_normal=False（全群は常に正規）")
        if sub.is_trivial:
            n_triv += 1
        if sub.is_whole_group:
            n_whole += 1
    if n_triv != 1:
        r.fail(f"is_trivial=True の部分群が {n_triv} 個（期待 1）")
    if n_whole != 1:
        r.fail(f"is_whole_group=True の部分群が {n_whole} 個（期待 1）")
    return r


def check_quotient_group_consistency(g: PointGroup) -> CheckResult:
    """quotient_group field consistency.

    - is_normal=True ⟹ quotient_group is not None
    - is_normal=False ⟹ quotient_group is None
    - 自明群 {E} の quotient_group は親群と同名（G/{E} ≅ G）
    - 全群 G の quotient_group は "C1"（G/G ≅ C1）
    """
    r = CheckResult("quotient_group_consistency")
    for sub in g.subgroups:
        if sub.is_normal and sub.quotient_group is None:
            r.fail(
                f"{sub.schoenflies}: 正規部分群なのに quotient_group が None"
            )
        if not sub.is_normal and sub.quotient_group is not None:
            r.fail(
                f"{sub.schoenflies}: 非正規なのに "
                f"quotient_group='{sub.quotient_group}' が設定されている"
            )
        if sub.is_trivial and sub.quotient_group != g.schoenflies:
            r.fail(
                f"{sub.schoenflies}: 自明群の quotient_group="
                f"'{sub.quotient_group}'（期待: '{g.schoenflies}'）"
            )
        if sub.is_whole_group and sub.quotient_group != "C1":
            r.fail(
                f"{sub.schoenflies}: 全群の quotient_group="
                f"'{sub.quotient_group}'（期待: 'C1'）"
            )
    return r


def check_coset_representatives(g: PointGroup) -> CheckResult:
    """coset_representatives の長さは index_in_parent と一致し、
    実際に左剰余類の代表系を成すこと。"""
    r = CheckResult("coset_representatives")
    for sub in g.subgroups:
        reps = sub.coset_representatives
        if len(reps) != sub.index_in_parent:
            r.fail(
                f"{sub.schoenflies}: |coset_representatives|={len(reps)} "
                f"≠ index_in_parent={sub.index_in_parent}"
            )
            continue
        # 各 rep × H が互いに素で、合わせて G になるか
        H = set(sub.element_indices)
        seen = set()
        for x in reps:
            coset = {g.multiply(x, h) for h in H}
            if coset & seen:
                r.fail(
                    f"{sub.schoenflies}: coset_representatives が重複した剰余類を含む "
                    f"(rep {x} の剰余類が他と交わる)"
                )
                break
            seen |= coset
        else:
            if seen != set(range(g.order)):
                missing = set(range(g.order)) - seen
                r.fail(
                    f"{sub.schoenflies}: coset_representatives が G を覆わない "
                    f"(欠けている要素: {sorted(missing)})"
                )
    return r


def check_is_maximal(g: PointGroup) -> CheckResult:
    """is_maximal flag must agree with the actual maximality test.

    A proper subgroup H is maximal iff there is no proper subgroup K
    with H < K ⊊ G.
    """
    r = CheckResult("is_maximal")
    proper = [
        s for s in g.subgroups
        if not s.is_trivial and not s.is_whole_group
    ]
    for s in proper:
        H = set(s.element_indices)
        actual_maximal = True
        for t in proper:
            if t is s:
                continue
            T = set(t.element_indices)
            if H < T:  # H が真の部分集合
                actual_maximal = False
                break
        if actual_maximal != s.is_maximal:
            r.fail(
                f"{s.schoenflies}: declared is_maximal={s.is_maximal}, "
                f"actual={actual_maximal}"
            )
    # 自明群と全群は is_maximal=False（慣習）
    for s in g.subgroups:
        if (s.is_trivial or s.is_whole_group) and s.is_maximal:
            r.fail(
                f"{s.schoenflies}: 自明群/全群は is_maximal=False のはず"
                f"（極大「真」部分群の意味）"
            )
    return r


def check_element_to_index(g: PointGroup) -> CheckResult:
    """raw JSON の element_to_index と elements[i].schoenflies が一致するか。"""
    r = CheckResult("element_to_index")
    raw = g._raw
    e2i = raw.get("element_to_index")
    if e2i is None:
        r.fail("element_to_index フィールドが存在しない")
        return r
    # 双方向一致を確認
    name_to_idx_from_elems = {e.schoenflies: e.index for e in g.elements}
    if dict(e2i) != name_to_idx_from_elems:
        # どちらにあって、どちらに無いかを詳しく報告
        only_in_e2i = set(e2i.items()) - set(name_to_idx_from_elems.items())
        only_in_elems = set(name_to_idx_from_elems.items()) - set(e2i.items())
        if only_in_e2i:
            r.fail(
                f"element_to_index にあるが elements と不一致: "
                f"{sorted(only_in_e2i)}"
            )
        if only_in_elems:
            r.fail(
                f"elements にあるが element_to_index と不一致: "
                f"{sorted(only_in_elems)}"
            )
    if len(e2i) != g.order:
        r.fail(f"|element_to_index|={len(e2i)} ≠ order={g.order}")
    return r


def check_generators_generate(g: PointGroup) -> CheckResult:
    """generators から閉包を取ったとき、群全体が生成されるか。

    C1（自明群）の場合は generators が空でよい（{E} 自体が全群）。
    """
    r = CheckResult("generators_generate")
    if not g.generators:
        # 自明群以外で generators が空なのはエラー
        if g.order != 1:
            r.fail(f"generators が空（order {g.order} 群では不正）")
        return r
    H = set(g.generators) | {0}
    changed = True
    while changed:
        changed = False
        new = set()
        for a in H:
            for b in H:
                p = g.multiply(a, b)
                if p not in H:
                    new.add(p)
        if new:
            H |= new
            changed = True
    if H != set(range(g.order)):
        missing = set(range(g.order)) - H
        r.fail(
            f"generators {g.generators} は群全体を生成しない "
            f"(生成サイズ {len(H)}, 欠けている要素: {sorted(missing)[:10]}"
            f"{'...' if len(missing) > 10 else ''})"
        )
    return r


def check_matrix_orthogonal(g: PointGroup, tol: float = 1e-9) -> CheckResult:
    """各要素の行列が直交行列であり、det = ±1 であるか。

    点群の元はすべて直交変換でなければならない。
    """
    r = CheckResult("matrix_orthogonal")
    eye = np.eye(3)
    for e in g.elements:
        M = e.matrix
        if not np.allclose(M.T @ M, eye, atol=tol):
            r.fail(
                f"g_{e.index} ({e.schoenflies}): 直交行列でない (M^T M ≠ I)"
            )
        d = float(np.linalg.det(M))
        if abs(abs(d) - 1) > tol:
            r.fail(f"g_{e.index} ({e.schoenflies}): |det| ≠ 1 (det={d})")
    return r


def check_type_matches_det(g: PointGroup, tol: float = 1e-9) -> CheckResult:
    """element の type と行列式の対応:
        identity / rotation              -> det = +1
        reflection / inversion / improper_rotation -> det = -1

    （improper_rotation は S_n = σh × C_n、つまり rotoinversion と同義。
    既存規約に合わせて 'improper_rotation' を採用）
    """
    r = CheckResult("type_matches_det")
    proper_types = {"identity", "rotation"}
    improper_types = {"reflection", "inversion", "improper_rotation"}
    for e in g.elements:
        d = round(float(np.linalg.det(e.matrix)))
        if e.type in proper_types and d != 1:
            r.fail(
                f"g_{e.index} ({e.schoenflies}): type={e.type} だが det={d}"
            )
        elif e.type in improper_types and d != -1:
            r.fail(
                f"g_{e.index} ({e.schoenflies}): type={e.type} だが det={d}"
            )
        elif e.type not in proper_types | improper_types:
            r.fail(
                f"g_{e.index} ({e.schoenflies}): 未知の type='{e.type}'"
            )
    return r


def check_axis_normalization(g: PointGroup, tol: float = 1e-9) -> CheckResult:
    """axis / plane_normal が単位ベクトル（ノルム 1）として書かれているか。"""
    r = CheckResult("axis_normalization")
    for e in g.elements:
        if e.axis is not None:
            norm = float(np.linalg.norm(e.axis))
            if abs(norm - 1.0) > tol:
                r.fail(
                    f"g_{e.index} ({e.schoenflies}): axis のノルム "
                    f"{norm:.6f} ≠ 1（{e.axis}）"
                )
        if e.plane_normal is not None:
            norm = float(np.linalg.norm(e.plane_normal))
            if abs(norm - 1.0) > tol:
                r.fail(
                    f"g_{e.index} ({e.schoenflies}): plane_normal のノルム "
                    f"{norm:.6f} ≠ 1（{e.plane_normal}）"
                )
    return r


def check_conjugacy_class_homogeneous(g: PointGroup) -> CheckResult:
    """同じ共役類の元はすべて同じ element_order を持つ（共役類の不変量）。"""
    r = CheckResult("conjugacy_class_homogeneous")
    for cc in g.conjugacy_classes:
        orders = {g.elements[i].element_order for i in cc}
        if len(orders) > 1:
            r.fail(
                f"共役類 {cc} に異なる element_order の元が混在: {sorted(orders)}"
            )
        # 共役類サイズは |G| を割る（軌道-安定子定理の系）
        if g.order % len(cc) != 0:
            r.fail(
                f"共役類 {cc} のサイズ {len(cc)} が order {g.order} を割らない"
            )
    return r


def check_special_subgroups_listed(g: PointGroup) -> CheckResult:
    """中心 Z(G) と交換子部分群 [G,G] は subgroups リストに必ず存在する。"""
    r = CheckResult("special_subgroups_listed")
    Z = sorted(g.center)
    C = sorted(g.commutator_subgroup)
    sub_sets = [sorted(s.element_indices) for s in g.subgroups]
    if Z not in sub_sets:
        r.fail(f"中心 Z(G) = {Z} が subgroups リストに無い")
    if C not in sub_sets:
        r.fail(f"交換子部分群 [G,G] = {C} が subgroups リストに無い")
    return r


def check_index_consistency() -> CheckResult:
    """data/index.json と各 JSON ファイルの整合性をチェック。

    - index.json で参照されるファイルが実在
    - data/ 内の XX_*.json ファイルが index.json に登録されている
    - id とファイル名 prefix が一致（id=N → "NN_*.json"）
    - 各 JSON の schoenflies / hermann_mauguin / order / crystal_system が
      index.json と一致
    """
    from loader import DATA_DIR
    r = CheckResult("index_consistency")
    idx_entries = list_groups()

    # ファイルの存在確認 + 双方向の比較
    idx_files = {e["file"] for e in idx_entries}
    actual_files = {p.name for p in DATA_DIR.glob("[0-9][0-9]_*.json")}
    missing = idx_files - actual_files
    extra = actual_files - idx_files
    if missing:
        r.fail(f"index.json に登録されているが data/ に無いファイル: {sorted(missing)}")
    if extra:
        r.fail(f"data/ にあるが index.json に登録されていないファイル: {sorted(extra)}")

    # id とファイル名 prefix
    seen_ids = set()
    for entry in idx_entries:
        if entry["id"] in seen_ids:
            r.fail(f"id={entry['id']} が重複している")
        seen_ids.add(entry["id"])
        prefix = f"{entry['id']:02d}_"
        if not entry["file"].startswith(prefix):
            r.fail(
                f"id={entry['id']} のファイル名 '{entry['file']}' が "
                f"prefix '{prefix}' で始まっていない"
            )

    # 各 JSON の中身と index.json のメタデータ
    for entry in idx_entries:
        try:
            g = load_group(entry["schoenflies"])
        except Exception as e:
            r.fail(f"{entry['schoenflies']}: 読み込み失敗 ({e})")
            continue
        if g.schoenflies != entry["schoenflies"]:
            r.fail(
                f"{entry['file']}: schoenflies 不一致 "
                f"(index='{entry['schoenflies']}', file='{g.schoenflies}')"
            )
        if g.hermann_mauguin != entry["hermann_mauguin"]:
            r.fail(
                f"{g.schoenflies}: hermann_mauguin 不一致 "
                f"(index='{entry['hermann_mauguin']}', file='{g.hermann_mauguin}')"
            )
        if g.order != entry["order"]:
            r.fail(
                f"{g.schoenflies}: order 不一致 "
                f"(index={entry['order']}, file={g.order})"
            )
        if g.crystal_system != entry["crystal_system"]:
            r.fail(
                f"{g.schoenflies}: crystal_system 不一致 "
                f"(index='{entry['crystal_system']}', file='{g.crystal_system}')"
            )
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
    check_inverse_symmetry,
    check_element_order,
    check_matrix_consistency,
    check_matrix_orthogonal,
    check_type_matches_det,
    check_axis_normalization,
    check_element_to_index,
    check_generators_generate,
    check_subgroup_identity,
    check_subgroup_order,
    check_subgroup_closure,
    check_subgroup_flags,
    check_quotient_group_consistency,
    check_coset_representatives,
    check_is_maximal,
    check_lagrange,
    check_normality_flags,
    check_conjugacy_classes,
    check_conjugacy_partition,
    check_conjugacy_class_homogeneous,
    check_abelian_flag,
    check_abelian_consequences,
    check_center,
    check_commutator_subgroup,
    check_special_subgroups_listed,
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

    # 全群モードのときだけ index.json のグローバル整合性をチェック
    if not args.groups:
        idx_res = check_index_consistency()
        if not idx_res.ok:
            mark = "NG"
            if not args.quiet:
                print(f"=== index.json 整合性 ===")
                print(f"  [{mark}] {idx_res.name}")
            else:
                print(f"[NG] (global): {idx_res.name}")
            for err in idx_res.errors[:10]:
                indent = "      " if not args.quiet else "    "
                for line in err.splitlines():
                    print(f"{indent}{line}")
            if len(idx_res.errors) > 10:
                print(f"      ... and {len(idx_res.errors) - 10} more errors")
            total_failures += 1
            if not args.quiet:
                print()
        elif not args.quiet:
            print("=== index.json 整合性 ===")
            print(f"  [OK] {idx_res.name}")
            print()

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
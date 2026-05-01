# Consistency Audit Report

## Scope

- Audited 32 point-group JSON files in `data/`.
- The audit script does not modify JSON files.
- The audit focuses on cross-file and semantic checks that go beyond the existing pytest/validate.py checks.

## Additional Checks

- Exhaustive subgroup enumeration from each multiplication table versus declared `subgroups`.
- Actual quotient group `G/H` for every declared normal subgroup versus the named `quotient_group`.
- Actual abelianization `G/[G,G]` versus the declared `abelianization` group.
- Matrix compatibility of geometric hints: rotation axes, reflection plane normals, and rotation angles.

## Findings

No inconsistencies were found by the additional audit.

## Summary

- Findings: 0
- Groups with findings: 0

## Reproduction

```bash
python3 scripts/audit_consistency.py --write-report audit_report.md
```

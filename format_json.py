"""
JSON 整形ツール: point_group リポジトリの既存規約に合わせて整形する。

ルール:
- 数値・文字列のみを含む「葉の配列」は 1 行にする
  例: [1.0, 0.0, 0.0], [0, 0, 0], [0, 1, 2, 3]
- 多重配列でも、内側がすべて葉配列なら、各内側配列が 1 行ずつ並ぶ
  例: matrix の 3x3 や multiplication_table
- オブジェクトの中身は基本通常通りインデント
- ただし conjugacy_classes のように「葉配列の配列」も各要素が 1 行

使い方:
    python format_json.py <file.json> [<file.json> ...]
    （引数なしなら data/*.json をすべて整形）
"""
from __future__ import annotations
import json
import sys
from pathlib import Path


def is_leaf_array(x):
    """要素がすべて int / float / str / bool / None / 葉配列 だけからなる配列か。"""
    if not isinstance(x, list):
        return False
    return all(
        isinstance(e, (int, float, str, bool)) or e is None
        for e in x
    )


def is_array_of_leaf_arrays(x):
    """配列で、要素がすべて葉配列のもの（multiplication_table, matrix）。"""
    if not isinstance(x, list):
        return False
    return all(is_leaf_array(e) for e in x)


def fmt(value, indent_level=0, indent_str="  "):
    """value を整形済み JSON 文字列として返す。"""
    pad = indent_str * indent_level
    pad_inner = indent_str * (indent_level + 1)

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)

    if isinstance(value, list):
        if len(value) == 0:
            return "[]"
        if is_leaf_array(value):
            # 一行表記
            inner = ", ".join(fmt(v, 0, indent_str) for v in value)
            return "[" + inner + "]"
        if is_array_of_leaf_arrays(value):
            # 各内側配列を 1 行ずつ
            lines = ["["]
            for i, v in enumerate(value):
                comma = "," if i < len(value) - 1 else ""
                lines.append(pad_inner + fmt(v, 0, indent_str) + comma)
            lines.append(pad + "]")
            return "\n".join(lines)
        # 一般配列
        lines = ["["]
        for i, v in enumerate(value):
            comma = "," if i < len(value) - 1 else ""
            lines.append(pad_inner + fmt(v, indent_level + 1, indent_str) + comma)
        lines.append(pad + "]")
        return "\n".join(lines)

    if isinstance(value, dict):
        if len(value) == 0:
            return "{}"
        lines = ["{"]
        items = list(value.items())
        for i, (k, v) in enumerate(items):
            comma = "," if i < len(items) - 1 else ""
            key = json.dumps(k, ensure_ascii=False)
            val = fmt(v, indent_level + 1, indent_str)
            lines.append(f"{pad_inner}{key}: {val}{comma}")
        lines.append(pad + "}")
        return "\n".join(lines)

    raise TypeError(f"unsupported type: {type(value)}")


def format_file(path: Path) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = fmt(raw, indent_level=0) + "\n"
    path.write_text(out, encoding="utf-8")
    print(f"  formatted: {path}")


def main():
    if len(sys.argv) > 1:
        targets = [Path(p) for p in sys.argv[1:]]
    else:
        # デフォルト: data/*.json
        here = Path(__file__).resolve().parent
        # スクリプトが /home/claude にある場合に備え、point_group/data も探す
        candidates = [
            here / "data",
            Path("/home/claude/point_group/data"),
        ]
        for d in candidates:
            if d.exists():
                targets = sorted(d.glob("[0-9][0-9]_*.json"))
                break
        else:
            print("data ディレクトリが見つかりません。引数で指定してください。",
                  file=sys.stderr)
            sys.exit(1)
    for path in targets:
        format_file(path)


if __name__ == "__main__":
    main()

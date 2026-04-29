# point_group

32 結晶点群の構造データ（積表・部分群・正規部分群）を JSON で管理するリポジトリ。

## ディレクトリ構成

```
point_group/
├── data/
│   ├── index.json              # 群一覧のメタデータ
│   └── XX_NAME.json            # 各点群のデータ
├── loader.py                    # データ読み込みライブラリ
├── validate.py                  # 整合性チェック
├── tests/
│   └── test_groups.py           # pytest 用
└── .github/workflows/validate.yml
```

## 使い方

### データの読み込み

```python
from loader import load_group

g = load_group("C3v")
print(g.order)                   # 6
print(g.multiply(1, 3))          # C3 * sigma_v のインデックス
print(g.normal_subgroups())      # 正規部分群リスト
```

### 検証

```bash
python validate.py               # 全群を検証
python validate.py C3v           # C3v だけ検証
pytest tests/                    # pytest で実行
```

## 規約

- 行列基底: 直交デカルト、主回転軸を z 軸に取る
- 乗積表: 左作用 `table[i][j] = k ⇔ M_i M_j = M_k`（g_j を先に作用）
- インデックス 0 は常に単位元

## ライセンス

MIT

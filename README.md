# profile-correspondence-encoder

`profile-correspondence-encoder` provides a scikit-learn compatible `PCEEncoder` for high-cardinality categorical data.

It is inspired by MCA/correspondence analysis, but fits on a sparse category co-occurrence graph and outputs dense latent vectors.

## Install

```bash
pip install profile-correspondence-encoder
```

## Why Use It

- Unsupervised (no target leakage when fit on train only)
- pandas-friendly
- No one-hot explosion in output dimensionality
- Scalable sparse fit over category graph
- Canonical normalization + optional aliases
- Frequency-based shrinkage for rare categories
- Streaming/chunked fit and customizable graph construction

## Quick Start

```python
import pandas as pd
from profile_correspondence_encoder import PCEEncoder

df = pd.DataFrame({
    "city": ["New York", "NY", "new-york", "Los Angeles", "LA", None],
    "state": ["New York", "New York", "New York", "California", "California", None],
    "segment": ["A", "A", "A", "B", "B", "B"],
})

encoder = PCEEncoder(
    columns=["city", "state", "segment"],
    n_components=3,
    min_frequency=2,
    aliases={"city": {"ny": "new york", "la": "los angeles"}},
    output_format="pandas",
)

X_enc = encoder.fit_transform(df)
print(X_enc.head())
print(encoder.get_metadata().head())
print(encoder.fit_stats_)
```

## Full Parameter Reference

`PCEEncoder` signature:

```python
PCEEncoder(
    columns=None,
    n_components=4,
    min_frequency=5,
    normalize_text=True,
    lowercase=True,
    strip_accents=True,
    separator=" ",
    rare_token="__RARE__",
    unknown_token="__UNK__",
    missing_token="__MISSING__",
    aliases=None,
    dtype=np.float32,
    svd_n_iter=7,
    random_state=42,
    output_format="numpy",
    enable_rare_shrinkage=True,
    shrinkage_lambda=5.0,
    pair_count_method="lexsort",
    chunk_size=None,
    max_categories_per_column=None,
    pairing_strategy="all",
    anchor_columns=None,
    pair_sample_ratio=1.0,
    n_jobs=1,
    edge_weighting="count",
    svd_algorithm="randomized",
)
```

### Core Representation

- `columns`: `list[str] | None`
  Columns to encode. If `None`, uses all input columns.
- `n_components`: `int` (default `4`)
  Latent dimensions per categorical column.
- `min_frequency`: `int` (default `5`)
  Minimum count for a category to keep a dedicated code.

### Text Canonicalization

- `normalize_text`: `bool` (default `True`)
  Enables text normalization pipeline.
- `lowercase`: `bool` (default `True`)
  Lowercases category strings.
- `strip_accents`: `bool` (default `True`)
  Removes accents/diacritics.
- `separator`: `str` (default `" "`)
  Separator used after punctuation/whitespace normalization.
- `aliases`: `dict[str, dict[str, str]] | None`
  Manual synonym mapping per column after normalization.

### Special Tokens

- `rare_token`: `str` (default `"__RARE__"`)
- `unknown_token`: `str` (default `"__UNK__"`)
- `missing_token`: `str` (default `"__MISSING__"`)

### Rare-Category Shrinkage

- `enable_rare_shrinkage`: `bool` (default `True`)
  Applies smooth interpolation for seen-rare categories.
- `shrinkage_lambda`: `float` (default `5.0`)
  Controls shrinkage strength with `alpha = n / (n + lambda)`.

### Graph Construction Performance

- `pair_count_method`: `{"lexsort", "unique"}` (default `"lexsort"`)
  Integer pair-count implementation. `lexsort` is usually faster.
- `chunk_size`: `int | None` (default `None`)
  If set, fit is processed in row chunks (streaming-like behavior).
- `max_categories_per_column`: `int | None` (default `None`)
  Optional top-K cap (after `min_frequency`) per column.
- `pairing_strategy`: `{"all", "anchor", "sample"}` (default `"all"`)
  How column pairs are connected in the graph.
- `anchor_columns`: `list[str] | None`
  Required when `pairing_strategy="anchor"`.
- `pair_sample_ratio`: `float` in `(0, 1]` (default `1.0`)
  Used when `pairing_strategy="sample"`.
- `n_jobs`: `int` (default `1`)
  Parallelism for per-pair counting.

### Edge Weighting / Decomposition

- `edge_weighting`: `{"count", "pmi", "ppmi"}` (default `"count"`)
  Edge transformation before spectral embedding.
- `svd_algorithm`: `{"randomized", "arpack"}` (default `"randomized"`)
  Truncated SVD backend.
- `svd_n_iter`: `int` (default `7`)
  Power iterations for SVD.
- `random_state`: `int | None` (default `42`)
  Seed for reproducibility.

### Output

- `output_format`: `{"numpy", "pandas"}` (default `"numpy"`)
  Output type for `transform`.
- `dtype`: numpy dtype (default `np.float32`)
  Embedding/output numeric dtype.

## Customization Recipes

### Baseline (Recommended)

```python
enc = PCEEncoder(
    columns=cat_cols,
    n_components=4,
    min_frequency=10,
    output_format="numpy",
)
```

### Large-Scale / Memory-Aware

```python
enc = PCEEncoder(
    columns=cat_cols,
    n_components=6,
    min_frequency=20,
    chunk_size=250_000,
    max_categories_per_column=100_000,
    pair_count_method="lexsort",
    n_jobs=4,
    output_format="numpy",
)
```

### Very Wide Tables (Reduce Pair Explosion)

```python
enc = PCEEncoder(
    columns=cat_cols,
    pairing_strategy="anchor",
    anchor_columns=["country", "segment", "device_type"],
    n_components=4,
)
```

### Faster Approximate Pair Graph

```python
enc = PCEEncoder(
    columns=cat_cols,
    pairing_strategy="sample",
    pair_sample_ratio=0.35,
    n_components=4,
    random_state=42,
)
```

### Quality-Oriented Weighting

```python
enc = PCEEncoder(
    columns=cat_cols,
    edge_weighting="ppmi",
    n_components=8,
    svd_n_iter=10,
)
```

### Rare Handling Control

```python
enc = PCEEncoder(
    columns=cat_cols,
    min_frequency=15,
    enable_rare_shrinkage=True,
    shrinkage_lambda=8.0,
)
```

## Scikit-learn Pipeline Example (Titanic)

```python
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from profile_correspondence_encoder import PCEEncoder

# Public Titanic dataset via OpenML
data = fetch_openml(name="titanic", version=1, as_frame=True)
X = data.data[["sex", "embarked", "pclass"]].copy()
y = (data.target == "1").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pipe = Pipeline(
    steps=[
        (
            "pce",
            PCEEncoder(
                columns=["sex", "embarked", "pclass"],
                n_components=4,
                min_frequency=10,
                output_format="numpy",
            ),
        ),
        ("clf", LogisticRegression(max_iter=2000)),
    ]
)

pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, proba))
```

## Useful Methods and Attributes

After fitting:

- `transform(X)`: transform new data
- `get_feature_names_out()`: output feature names
- `get_metadata()`: raw/canonical/count/code metadata
- `get_column_embedding(column)`: embedding table for one column
- `fit_stats_`: fit timing metrics (for monitoring regressions)

## Development

```bash
pip install -e .[dev]
pytest
python -m build
```

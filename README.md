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
- Frequency-based shrinkage for rare categories (instead of hard bucketing)

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
```

## Scikit-learn Pipeline Example (Titanic)

```python
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from profile_correspondence_encoder import PCEEncoder

# Public Titanic dataset via OpenML
data = fetch_openml(name="titanic", version=1, as_frame=True)
X = data.data
y = (data.target == "1").astype(int)

cat_cols = ["sex", "embarked", "pclass"]
num_cols = ["age", "fare", "sibsp", "parch"]

preprocess = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "pce",
                        PCEEncoder(
                            columns=cat_cols,
                            n_components=4,
                            min_frequency=10,
                            output_format="numpy",
                        ),
                    ),
                ]
            ),
            cat_cols,
        ),
        (
            "num",
            Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
            num_cols,
        ),
    ],
    sparse_threshold=0.0,
)

model = Pipeline(
    steps=[
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=2000)),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, proba))
```

## Rare Category Shrinkage

Instead of mapping every rare category to the exact same vector, `PCEEncoder` shrinks seen-rare categories according to their fit frequency:

- `count >= min_frequency`: direct category vector
- `0 < count < min_frequency`: smooth interpolation between `__UNK__` and `__RARE__`
- unseen category: `__UNK__`

This keeps behavior stable while preserving some frequency signal among rare labels.

## Development

```bash
pip install -e .[dev]
pytest
python -m build
```

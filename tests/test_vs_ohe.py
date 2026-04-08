import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from profile_correspondence_encoder import PCEEncoder


def _make_variant_shift_dataset(n_rows: int = 5000, seed: int = 7):
    rng = np.random.default_rng(seed)

    city_levels = ["new york", "los angeles", "chicago", "miami"]
    state_map = {
        "new york": "new york",
        "los angeles": "california",
        "chicago": "illinois",
        "miami": "florida",
    }
    city_risk = {
        "new york": 0.80,
        "los angeles": 0.65,
        "chicago": 0.35,
        "miami": 0.20,
    }

    city_variants_train = {
        "new york": "new york",
        "los angeles": "los angeles",
        "chicago": "chicago",
        "miami": "miami",
    }
    city_variants_test = {
        "new york": " NEW-YORK ",
        "los angeles": "LOS_ANGELES",
        "chicago": "chi-ca-go",
        "miami": "M I A M I",
    }

    state_variants_train = {
        "new york": "new york",
        "california": "california",
        "illinois": "illinois",
        "florida": "florida",
    }
    state_variants_test = {
        "new york": "NEW_YORK",
        "california": "CALI-FORNIA",
        "illinois": "ILLI  NOIS",
        "florida": "FLO-RIDA",
    }

    city = rng.choice(city_levels, size=n_rows, p=[0.28, 0.26, 0.24, 0.22])
    state = np.array([state_map[c] for c in city], dtype=object)
    y = np.array([rng.random() < city_risk[c] for c in city], dtype=int)

    # deterministic split by index; test receives only unseen string variants
    split = int(n_rows * 0.75)
    city_str = np.empty(n_rows, dtype=object)
    state_str = np.empty(n_rows, dtype=object)

    for i, c in enumerate(city):
        s = state[i]
        if i < split:
            city_str[i] = city_variants_train[c]
            state_str[i] = state_variants_train[s]
        else:
            city_str[i] = city_variants_test[c]
            state_str[i] = state_variants_test[s]

    X = pd.DataFrame({"city": city_str, "state": state_str, "segment": rng.choice(["a", "b"], size=n_rows)})

    X_train = X.iloc[:split].reset_index(drop=True)
    y_train = y[:split]
    X_test = X.iloc[split:].reset_index(drop=True)
    y_test = y[split:]

    return X_train, X_test, y_train, y_test


def test_pce_beats_ohe_on_variant_shift():
    X_train, X_test, y_train, y_test = _make_variant_shift_dataset()

    pce = Pipeline(
        steps=[
            (
                "enc",
                PCEEncoder(
                    columns=["city", "state", "segment"],
                    n_components=4,
                    min_frequency=10,
                    output_format="numpy",
                    random_state=42,
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    ohe = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ["city", "state", "segment"],
                        )
                    ],
                    sparse_threshold=0.0,
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    pce.fit(X_train, y_train)
    ohe.fit(X_train, y_train)

    auc_pce = roc_auc_score(y_test, pce.predict_proba(X_test)[:, 1])
    auc_ohe = roc_auc_score(y_test, ohe.predict_proba(X_test)[:, 1])

    # PCE should improve under string-variant shift due canonical normalization.
    assert auc_pce >= auc_ohe + 0.02


def test_pce_not_worse_than_ohe_on_standard_split_titanic():
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name="titanic", version=1, as_frame=True)
    X = data.data[["sex", "embarked", "pclass"]].copy()
    y = (data.target == "1").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pce = Pipeline(
        steps=[
            (
                "enc",
                PCEEncoder(
                    columns=["sex", "embarked", "pclass"],
                    n_components=4,
                    min_frequency=10,
                    output_format="numpy",
                    random_state=42,
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    ohe = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ["sex", "embarked", "pclass"],
                        )
                    ],
                    sparse_threshold=0.0,
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    pce.fit(X_train, y_train)
    ohe.fit(X_train, y_train)

    auc_pce = roc_auc_score(y_test, pce.predict_proba(X_test)[:, 1])
    auc_ohe = roc_auc_score(y_test, ohe.predict_proba(X_test)[:, 1])

    # On regular split, require near-parity (small tolerance).
    assert auc_pce >= auc_ohe - 0.02

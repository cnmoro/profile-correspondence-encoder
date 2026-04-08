import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from profile_correspondence_encoder import PCEEncoder


def _make_multiclass_variant_shift_dataset(n_rows: int = 7000, seed: int = 13):
    rng = np.random.default_rng(seed)

    city_levels = ["new york", "los angeles", "chicago", "miami"]
    state_map = {
        "new york": "new york",
        "los angeles": "california",
        "chicago": "illinois",
        "miami": "florida",
    }

    city_probs = {
        "new york": np.array([0.75, 0.20, 0.05]),
        "los angeles": np.array([0.20, 0.70, 0.10]),
        "chicago": np.array([0.15, 0.25, 0.60]),
        "miami": np.array([0.45, 0.25, 0.30]),
    }

    seg_levels = ["a", "b", "c"]
    seg_shift = {
        "a": np.array([0.05, -0.02, -0.03]),
        "b": np.array([-0.03, 0.06, -0.03]),
        "c": np.array([-0.02, -0.01, 0.03]),
    }

    city_train = {
        "new york": "new york",
        "los angeles": "los angeles",
        "chicago": "chicago",
        "miami": "miami",
    }
    city_test = {
        "new york": " NEW-YORK ",
        "los angeles": "LOS_ANGELES",
        "chicago": "chi-ca-go",
        "miami": "M I A M I",
    }

    state_train = {
        "new york": "new york",
        "california": "california",
        "illinois": "illinois",
        "florida": "florida",
    }
    state_test = {
        "new york": "NEW_YORK",
        "california": "CALI-FORNIA",
        "illinois": "ILLI  NOIS",
        "florida": "FLO-RIDA",
    }

    city = rng.choice(city_levels, size=n_rows, p=[0.28, 0.26, 0.24, 0.22])
    segment = rng.choice(seg_levels, size=n_rows, p=[0.35, 0.35, 0.30])
    device = rng.choice(["mobile", "desktop", "tablet"], size=n_rows, p=[0.6, 0.3, 0.1])

    y = np.empty(n_rows, dtype=int)
    for i, c in enumerate(city):
        p = city_probs[c] + seg_shift[segment[i]]
        p = np.clip(p, 1e-6, None)
        p = p / p.sum()
        y[i] = rng.choice(3, p=p)

    split = int(n_rows * 0.75)
    city_str = np.empty(n_rows, dtype=object)
    state_str = np.empty(n_rows, dtype=object)

    for i, c in enumerate(city):
        s = state_map[c]
        if i < split:
            city_str[i] = city_train[c]
            state_str[i] = state_train[s]
        else:
            city_str[i] = city_test[c]
            state_str[i] = state_test[s]

    X = pd.DataFrame(
        {
            "city": city_str,
            "state": state_str,
            "segment": segment,
            "device": device,
        }
    )

    x_train = X.iloc[:split].reset_index(drop=True)
    x_test = X.iloc[split:].reset_index(drop=True)
    y_train = y[:split]
    y_test = y[split:]

    return x_train, x_test, y_train, y_test


def _mk_pce_pipeline(model):
    cols = ["city", "state", "segment", "device"]
    return Pipeline(
        [
            (
                "enc",
                PCEEncoder(
                    columns=cols,
                    n_components=6,
                    min_frequency=10,
                    output_format="numpy",
                    random_state=42,
                ),
            ),
            ("clf", model),
        ]
    )


def _mk_ohe_pipeline(model):
    cols = ["city", "state", "segment", "device"]
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            cols,
                        )
                    ],
                    sparse_threshold=0.0,
                ),
            ),
            ("clf", model),
        ]
    )


def test_multiclass_logreg_pce_beats_ohe_on_variant_shift():
    x_train, x_test, y_train, y_test = _make_multiclass_variant_shift_dataset()

    pce = _mk_pce_pipeline(LogisticRegression(max_iter=3000))
    ohe = _mk_ohe_pipeline(LogisticRegression(max_iter=3000))

    pce.fit(x_train, y_train)
    ohe.fit(x_train, y_train)

    y_pred_pce = pce.predict(x_test)
    y_pred_ohe = ohe.predict(x_test)

    acc_pce = accuracy_score(y_test, y_pred_pce)
    acc_ohe = accuracy_score(y_test, y_pred_ohe)
    f1_pce = f1_score(y_test, y_pred_pce, average="macro")
    f1_ohe = f1_score(y_test, y_pred_ohe, average="macro")

    assert acc_pce >= acc_ohe + 0.03
    assert f1_pce >= f1_ohe + 0.03


def test_multiclass_random_forest_pce_beats_ohe_on_variant_shift():
    x_train, x_test, y_train, y_test = _make_multiclass_variant_shift_dataset(seed=29)

    pce = _mk_pce_pipeline(RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
    ohe = _mk_ohe_pipeline(RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))

    pce.fit(x_train, y_train)
    ohe.fit(x_train, y_train)

    y_pred_pce = pce.predict(x_test)
    y_pred_ohe = ohe.predict(x_test)

    acc_pce = accuracy_score(y_test, y_pred_pce)
    acc_ohe = accuracy_score(y_test, y_pred_ohe)
    f1_pce = f1_score(y_test, y_pred_pce, average="macro")
    f1_ohe = f1_score(y_test, y_pred_ohe, average="macro")

    assert acc_pce >= acc_ohe + 0.02
    assert f1_pce >= f1_ohe + 0.02

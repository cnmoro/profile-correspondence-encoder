import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from profile_correspondence_encoder import PCEEncoder


def _make_small_df(seed: int = 0, n: int = 600):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.choice(["x", "y", "z", "x!", "X"], size=n),
            "b": rng.choice(["u", "v", "w", "w#"], size=n),
            "c": rng.choice(["m", "n", "o"], size=n),
            "d": rng.choice(["p", "q"], size=n),
        }
    )


def _gram(x: np.ndarray) -> np.ndarray:
    return x @ x.T


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
    state = np.array([state_map[c] for c in city], dtype=object)
    y = np.array([rng.random() < city_risk[c] for c in city], dtype=int)

    split = int(n_rows * 0.75)
    city_str = np.empty(n_rows, dtype=object)
    state_str = np.empty(n_rows, dtype=object)

    for i, c in enumerate(city):
        s = state[i]
        if i < split:
            city_str[i] = city_train[c]
            state_str[i] = state_train[s]
        else:
            city_str[i] = city_test[c]
            state_str[i] = state_test[s]

    X = pd.DataFrame({"city": city_str, "state": state_str, "segment": rng.choice(["a", "b"], size=n_rows)})
    return X.iloc[:split].reset_index(drop=True), X.iloc[split:].reset_index(drop=True), y[:split], y[split:]


def _mk_model(enc: PCEEncoder):
    return Pipeline([("enc", enc), ("clf", LogisticRegression(max_iter=2000))])


def _mk_ohe_model(cols):
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
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )


def test_pair_count_methods_match():
    df = _make_small_df()

    a = PCEEncoder(columns=["a", "b", "c", "d"], n_components=3, pair_count_method="unique", random_state=7)
    b = PCEEncoder(columns=["a", "b", "c", "d"], n_components=3, pair_count_method="lexsort", random_state=7)

    xa = a.fit_transform(df)
    xb = b.fit_transform(df)

    assert np.allclose(_gram(xa), _gram(xb), atol=1e-5)


def test_chunked_fit_matches_non_chunked():
    df = _make_small_df()

    a = PCEEncoder(columns=["a", "b", "c", "d"], n_components=3, chunk_size=None, random_state=3)
    b = PCEEncoder(columns=["a", "b", "c", "d"], n_components=3, chunk_size=73, random_state=3)

    xa = a.fit_transform(df)
    xb = b.fit_transform(df)

    assert np.allclose(_gram(xa), _gram(xb), atol=1e-5)


def test_integer_pipeline_codes_are_int32():
    df = _make_small_df()
    enc = PCEEncoder(columns=["a", "b", "c", "d"], n_components=2).fit(df)

    s = enc._prepare_series(df["a"], "a")
    codes, _, _ = enc._encode_series_for_fit_or_transform(s, "a")
    assert codes.dtype == np.int32


def test_topk_vocab_cap_applied():
    df = pd.DataFrame(
        {
            "a": [f"v{i}" for i in range(30)] * 3,
            "b": ["x"] * 90,
            "c": ["y"] * 90,
        }
    )

    enc = PCEEncoder(columns=["a", "b", "c"], n_components=2, min_frequency=1, max_categories_per_column=5)
    enc.fit(df)

    vocab_a = enc.vocab_["a"].tolist()
    assert len(vocab_a) <= 8
    assert "__RARE__" in vocab_a
    assert "__UNK__" in vocab_a


def test_anchor_pairing_reduces_pair_count():
    df = _make_small_df()
    enc = PCEEncoder(
        columns=["a", "b", "c", "d"],
        n_components=2,
        pairing_strategy="anchor",
        anchor_columns=["a"],
    ).fit(df)

    assert len(enc.pairs_) == 3
    assert all("a" in p for p in enc.pairs_)


def test_sample_pairing_is_deterministic_for_seed():
    df = _make_small_df()

    e1 = PCEEncoder(
        columns=["a", "b", "c", "d"],
        n_components=2,
        pairing_strategy="sample",
        pair_sample_ratio=0.5,
        random_state=11,
    ).fit(df)

    e2 = PCEEncoder(
        columns=["a", "b", "c", "d"],
        n_components=2,
        pairing_strategy="sample",
        pair_sample_ratio=0.5,
        random_state=11,
    ).fit(df)

    assert e1.pairs_ == e2.pairs_
    assert len(e1.pairs_) == 3


def test_parallel_pair_counting_matches_single_thread():
    df = _make_small_df()

    s1 = PCEEncoder(columns=["a", "b", "c", "d"], n_components=3, n_jobs=1, random_state=19)
    s2 = PCEEncoder(columns=["a", "b", "c", "d"], n_components=3, n_jobs=2, random_state=19)

    x1 = s1.fit_transform(df)
    x2 = s2.fit_transform(df)

    assert np.allclose(_gram(x1), _gram(x2), atol=1e-5)


def test_weighting_modes_are_stable_and_finite():
    df = _make_small_df(n=500)

    for w in ["count", "pmi", "ppmi"]:
        enc = PCEEncoder(columns=["a", "b", "c", "d"], n_components=3, edge_weighting=w, random_state=5)
        x = enc.fit_transform(df)
        assert np.isfinite(x).all()


def test_new_defaults_not_worse_than_legacy_like_baseline_and_beats_ohe():
    x_train, x_test, y_train, y_test = _make_variant_shift_dataset()
    cols = ["city", "state", "segment"]

    legacy_like = _mk_model(
        PCEEncoder(
            columns=cols,
            n_components=4,
            min_frequency=10,
            pair_count_method="unique",
            chunk_size=None,
            max_categories_per_column=None,
            pairing_strategy="all",
            edge_weighting="count",
            n_jobs=1,
            random_state=42,
        )
    )

    improved = _mk_model(
        PCEEncoder(
            columns=cols,
            n_components=4,
            min_frequency=10,
            random_state=42,
        )
    )

    ohe = _mk_ohe_model(cols)

    legacy_like.fit(x_train, y_train)
    improved.fit(x_train, y_train)
    ohe.fit(x_train, y_train)

    auc_legacy = roc_auc_score(y_test, legacy_like.predict_proba(x_test)[:, 1])
    auc_improved = roc_auc_score(y_test, improved.predict_proba(x_test)[:, 1])
    auc_ohe = roc_auc_score(y_test, ohe.predict_proba(x_test)[:, 1])

    t_legacy = legacy_like.named_steps["enc"].fit_stats_["fit_total_time_s"]
    t_improved = improved.named_steps["enc"].fit_stats_["fit_total_time_s"]

    assert auc_improved >= auc_legacy - 0.005
    assert auc_improved >= auc_ohe + 0.02
    assert t_improved <= 1.5 * t_legacy

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from profile_correspondence_encoder import PCEEncoder


def test_basic_fit_transform_shapes_and_feature_names():
    df = pd.DataFrame(
        {
            "city": ["New York", "NY", "new-york", "Los Angeles", "LA", None],
            "state": ["New York", "New York", "New York", "California", "California", None],
            "segment": ["A", "A", "A", "B", "B", "B"],
        }
    )

    enc = PCEEncoder(
        columns=["city", "state", "segment"],
        n_components=3,
        min_frequency=2,
        aliases={"city": {"ny": "new york", "la": "los angeles"}},
        output_format="numpy",
        random_state=42,
    )

    Xt = enc.fit_transform(df)
    assert Xt.shape == (len(df), 9)

    names = enc.get_feature_names_out()
    assert len(names) == 9
    assert names[0] == "city_pce_0"


def test_frequency_shrinkage_differs_from_hard_bucket_for_rare_seen_values():
    train = pd.DataFrame(
        {
            "c1": ["a", "a", "b", "c", "d", "e", "f", "f", "f"],
            "c2": ["x", "x", "y", "y", "z", "z", "x", "y", "z"],
            "c3": ["u", "u", "u", "v", "v", "w", "w", "w", "w"],
        }
    )

    test = pd.DataFrame(
        {
            "c1": ["b", "c", "new_value"],
            "c2": ["y", "z", "x"],
            "c3": ["u", "v", "w"],
        }
    )

    # min_frequency=3 => 'b' and 'c' are seen-rare, 'new_value' unseen
    enc = PCEEncoder(
        columns=["c1", "c2", "c3"],
        n_components=2,
        min_frequency=3,
        output_format="numpy",
        enable_rare_shrinkage=True,
        random_state=0,
    )
    enc.fit(train)
    xt = enc.transform(test)

    # b(count=1) and c(count=1) should have same shrunk c1 vector, but differ from unseen
    c1_block = xt[:, :2]
    assert np.allclose(c1_block[0], c1_block[1])
    assert not np.allclose(c1_block[0], c1_block[2])


def test_sklearn_pipeline_with_titanic_openml_runs():
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
                    random_state=42,
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    assert 0.6 <= auc <= 1.0

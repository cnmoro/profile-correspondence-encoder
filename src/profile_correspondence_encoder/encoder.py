from __future__ import annotations

import unicodedata
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.validation import check_is_fitted


class PCEEncoder(BaseEstimator, TransformerMixin):
    """Profile Correspondence Encoder (PCE).

    Scalable categorical encoder inspired by MCA/correspondence structure,
    implemented as a sparse category co-occurrence embedding.

    Frequency-based shrinkage:
    - categories with count >= min_frequency use their direct fitted vector.
    - seen categories with 0 < count < min_frequency are smoothly shrunk toward
      the rare-token vector using alpha = count / min_frequency.
    - unseen categories map to unknown-token vector.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        n_components: int = 4,
        min_frequency: int = 5,
        normalize_text: bool = True,
        lowercase: bool = True,
        strip_accents: bool = True,
        separator: str = " ",
        rare_token: str = "__RARE__",
        unknown_token: str = "__UNK__",
        missing_token: str = "__MISSING__",
        aliases: dict[str, dict[str, str]] | None = None,
        dtype: Any = np.float32,
        svd_n_iter: int = 7,
        random_state: int | None = 42,
        output_format: str = "numpy",
        enable_rare_shrinkage: bool = True,
    ) -> None:
        self.columns = columns
        self.n_components = int(n_components)
        self.min_frequency = int(min_frequency)
        self.normalize_text = normalize_text
        self.lowercase = lowercase
        self.strip_accents = strip_accents
        self.separator = separator
        self.rare_token = rare_token
        self.unknown_token = unknown_token
        self.missing_token = missing_token
        self.aliases = aliases
        self.dtype = dtype
        self.svd_n_iter = int(svd_n_iter)
        self.random_state = random_state
        self.output_format = output_format
        self.enable_rare_shrinkage = bool(enable_rare_shrinkage)

    def _validate_params(self) -> None:
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")
        if self.min_frequency < 1:
            raise ValueError("min_frequency must be >= 1")
        if self.output_format not in {"numpy", "pandas"}:
            raise ValueError("output_format must be 'numpy' or 'pandas'")

    def _normalize_series(self, s: pd.Series) -> pd.Series:
        s = s.copy().astype("string")
        s = s.where(~s.isna(), self.missing_token)

        if not self.normalize_text:
            return s

        if self.lowercase:
            s = s.str.lower()

        if self.strip_accents:
            s = s.map(
                lambda x: "".join(
                    ch
                    for ch in unicodedata.normalize("NFKD", str(x))
                    if not unicodedata.combining(ch)
                )
            )

        s = s.str.replace(r"[^a-zA-Z0-9]+", " ", regex=True)
        s = s.str.replace(r"\s+", self.separator, regex=True)
        s = s.str.strip()

        s = s.where(s != "", self.missing_token)
        return s

    def _build_alias_maps(self) -> dict[str, dict[str, str]]:
        alias_maps: dict[str, dict[str, str]] = {}
        if not self.aliases:
            return alias_maps

        for col, mapping in self.aliases.items():
            normalized_mapping: dict[str, str] = {}
            raw_keys = pd.Series(list(mapping.keys()), dtype="string")
            raw_vals = pd.Series(list(mapping.values()), dtype="string")

            norm_keys = self._normalize_series(raw_keys)
            norm_vals = self._normalize_series(raw_vals)

            for k, v in zip(norm_keys.tolist(), norm_vals.tolist()):
                normalized_mapping[k] = v

            alias_maps[col] = normalized_mapping

        return alias_maps

    def _apply_aliases(self, s: pd.Series, col: str) -> pd.Series:
        alias_map = self.alias_maps_.get(col)
        if not alias_map:
            return s
        return s.map(lambda x: alias_map.get(x, x))

    def _resolve_columns(self, X: pd.DataFrame) -> list[str]:
        if self.columns is None:
            return list(X.columns)
        return list(self.columns)

    def _prepare_fit_series(self, X: pd.DataFrame, col: str) -> pd.Series:
        s = self._normalize_series(X[col])
        s = self._apply_aliases(s, col)
        return s

    def _build_column_vocab(self, s: pd.Series):
        counts = s.value_counts(dropna=False)
        frequent = counts[counts >= self.min_frequency].index.tolist()

        if self.rare_token not in frequent:
            frequent.append(self.rare_token)
        if self.unknown_token not in frequent:
            frequent.append(self.unknown_token)
        if self.missing_token not in frequent:
            frequent.append(self.missing_token)

        vocab = pd.Index(pd.unique(pd.Series(frequent, dtype="string")))
        vocab_to_code = {v: i for i, v in enumerate(vocab.tolist())}

        return counts, vocab, vocab_to_code

    def _bucket_rare_for_fit(self, s: pd.Series, counts: pd.Series) -> pd.Series:
        frequent_mask = s.map(counts).fillna(0).astype(np.int64) >= self.min_frequency
        return s.where(frequent_mask, self.rare_token)

    def _series_to_local_codes_fit(self, s: pd.Series, vocab_to_code: dict[str, int]) -> np.ndarray:
        codes = s.map(vocab_to_code).fillna(vocab_to_code[self.unknown_token]).astype(np.int32)
        return codes.to_numpy()

    def _normalize_and_alias_column(self, X_col: pd.Series, col: str) -> pd.Series:
        s = self._normalize_series(X_col)
        s = self._apply_aliases(s, col)
        return s

    def _build_global_structure(self, local_codes_by_col: dict[str, np.ndarray]) -> sparse.csr_matrix:
        offsets: dict[str, int] = {}
        total_nodes = 0
        for col in self.columns_:
            offsets[col] = total_nodes
            total_nodes += len(self.vocab_[col])

        self.offsets_ = offsets
        self.total_nodes_ = total_nodes

        rows_all = []
        cols_all = []
        data_all = []

        total_nodes_i64 = np.int64(total_nodes)

        for col_a, col_b in combinations(self.columns_, 2):
            codes_a = local_codes_by_col[col_a]
            codes_b = local_codes_by_col[col_b]

            global_a = codes_a.astype(np.int64) + np.int64(offsets[col_a])
            global_b = codes_b.astype(np.int64) + np.int64(offsets[col_b])

            compressed = global_a * total_nodes_i64 + global_b
            uniq, cnt = np.unique(compressed, return_counts=True)

            pair_rows = (uniq // total_nodes_i64).astype(np.int32)
            pair_cols = (uniq % total_nodes_i64).astype(np.int32)
            pair_cnt = cnt.astype(self.dtype, copy=False)

            rows_all.append(pair_rows)
            cols_all.append(pair_cols)
            data_all.append(pair_cnt)

            rows_all.append(pair_cols)
            cols_all.append(pair_rows)
            data_all.append(pair_cnt)

        if not rows_all:
            raise ValueError("PCEEncoder requires at least 2 categorical columns.")

        rows = np.concatenate(rows_all)
        cols = np.concatenate(cols_all)
        data = np.concatenate(data_all)

        graph = sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(total_nodes, total_nodes),
            dtype=self.dtype,
        ).tocsr()
        graph.sum_duplicates()

        return graph

    def _fit_embedding(self, graph: sparse.csr_matrix) -> np.ndarray:
        degree = np.asarray(graph.sum(axis=1)).ravel().astype(np.float64)
        degree = np.maximum(degree, 1.0)

        inv_sqrt_degree = 1.0 / np.sqrt(degree)
        d_inv_sqrt = sparse.diags(inv_sqrt_degree)

        normalized_graph = d_inv_sqrt @ graph @ d_inv_sqrt

        k = min(self.n_components, max(1, normalized_graph.shape[0] - 1))

        svd = TruncatedSVD(
            n_components=k,
            n_iter=self.svd_n_iter,
            random_state=self.random_state,
        )
        embedding = svd.fit_transform(normalized_graph).astype(self.dtype, copy=False)

        if k < self.n_components:
            pad = np.zeros((embedding.shape[0], self.n_components - k), dtype=self.dtype)
            embedding = np.hstack([embedding, pad])

        return embedding

    def fit(self, X, y=None):
        self._validate_params()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.columns_ = self._resolve_columns(X)
        if len(self.columns_) < 2:
            raise ValueError("PCEEncoder requires at least 2 columns.")

        self.alias_maps_ = self._build_alias_maps()

        self.counts_ = {}
        self.vocab_ = {}
        self.vocab_to_code_ = {}
        self.local_code_to_value_ = {}
        self.raw_to_canonical_ = {}

        local_codes_by_col = {}

        for col in self.columns_:
            s_norm = self._prepare_fit_series(X, col)

            raw_strings = X[col].where(~X[col].isna(), self.missing_token).astype("string")
            raw_norm = self._normalize_series(raw_strings)
            raw_norm = self._apply_aliases(raw_norm, col)

            raw_map_df = pd.DataFrame(
                {"raw_value": raw_strings, "canonical_value": raw_norm},
                copy=False,
            ).drop_duplicates()
            self.raw_to_canonical_[col] = raw_map_df

            counts, vocab, vocab_to_code = self._build_column_vocab(s_norm)
            s_bucketed = self._bucket_rare_for_fit(s_norm, counts)
            codes = self._series_to_local_codes_fit(s_bucketed, vocab_to_code)

            self.counts_[col] = counts
            self.vocab_[col] = vocab
            self.vocab_to_code_[col] = vocab_to_code
            self.local_code_to_value_[col] = np.array(vocab.tolist(), dtype=object)
            local_codes_by_col[col] = codes

        graph = self._build_global_structure(local_codes_by_col)
        self.embedding_ = self._fit_embedding(graph)

        self.column_embedding_tables_ = {}
        self.column_default_vectors_ = {}
        self.column_rare_vectors_ = {}

        for col in self.columns_:
            start = self.offsets_[col]
            end = start + len(self.vocab_[col])

            table = self.embedding_[start:end]
            self.column_embedding_tables_[col] = table

            unk_code = self.vocab_to_code_[col][self.unknown_token]
            rare_code = self.vocab_to_code_[col][self.rare_token]

            self.column_default_vectors_[col] = table[unk_code]
            self.column_rare_vectors_[col] = table[rare_code]

        self.output_feature_names_ = [
            f"{col}_pce_{i}" for col in self.columns_ for i in range(self.n_components)
        ]
        self.n_features_in_ = len(self.columns_)

        return self

    def _transform_column_to_vectors(self, X_col: pd.Series, col: str) -> np.ndarray:
        s = self._normalize_and_alias_column(X_col, col)

        counts = self.counts_[col]
        vocab_to_code = self.vocab_to_code_[col]
        table = self.column_embedding_tables_[col]

        known_count = s.map(counts).fillna(0).astype(np.int64)
        s_bucketed = s.where(known_count >= self.min_frequency, self.rare_token)

        codes = s_bucketed.map(vocab_to_code).fillna(vocab_to_code[self.unknown_token]).astype(np.int32)
        codes_arr = np.clip(codes.to_numpy(), 0, table.shape[0] - 1)

        vectors = table[codes_arr].astype(self.dtype, copy=True)

        if not self.enable_rare_shrinkage:
            return vectors

        rare_code = vocab_to_code[self.rare_token]
        unk_code = vocab_to_code[self.unknown_token]
        rare_vec = table[rare_code]
        unk_vec = table[unk_code]

        specials = {self.rare_token, self.unknown_token, self.missing_token}
        is_shrink_candidate = (known_count.to_numpy() > 0) & (known_count.to_numpy() < self.min_frequency)
        if specials:
            is_special = s.isin(specials).to_numpy()
            is_shrink_candidate = is_shrink_candidate & ~is_special

        if np.any(is_shrink_candidate):
            alpha = (known_count.to_numpy(dtype=np.float64) / float(self.min_frequency)).clip(0.0, 1.0)
            alpha = alpha.astype(self.dtype, copy=False)
            idx = np.where(is_shrink_candidate)[0]
            a = alpha[idx][:, None]
            vectors[idx] = a * rare_vec + (1.0 - a) * unk_vec

        return vectors

    def transform(self, X):
        check_is_fitted(self, attributes=["embedding_", "columns_"])

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns_)

        n_rows = len(X)
        n_features = len(self.columns_) * self.n_components
        out = np.empty((n_rows, n_features), dtype=self.dtype)

        for j, col in enumerate(self.columns_):
            vectors = self._transform_column_to_vectors(X[col], col)
            start = j * self.n_components
            end = start + self.n_components
            out[:, start:end] = vectors

        if self.output_format == "pandas":
            return pd.DataFrame(out, columns=self.output_feature_names_, index=X.index)

        return out

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, attributes=["output_feature_names_"])
        return np.array(self.output_feature_names_, dtype=object)

    def get_metadata(self) -> pd.DataFrame:
        check_is_fitted(self, attributes=["embedding_", "columns_"])

        frames = []
        for col in self.columns_:
            vocab = self.vocab_[col]
            counts = self.counts_[col]

            fitted_df = pd.DataFrame(
                {
                    "column": col,
                    "canonical_value": vocab.astype("string"),
                    "count_in_fit": [int(counts.get(v, 0)) for v in vocab.tolist()],
                    "local_code": np.arange(len(vocab), dtype=np.int32),
                    "shrinkage_alpha": [
                        float(min(int(counts.get(v, 0)), self.min_frequency) / self.min_frequency)
                        if v not in {self.rare_token, self.unknown_token, self.missing_token}
                        else np.nan
                        for v in vocab.tolist()
                    ],
                }
            )

            raw_map_df = self.raw_to_canonical_[col].copy()
            raw_map_df["column"] = col

            merged = raw_map_df.merge(
                fitted_df,
                on=["column", "canonical_value"],
                how="left",
            )
            frames.append(merged)

        return pd.concat(frames, ignore_index=True)

    def get_column_embedding(self, column: str) -> pd.DataFrame:
        check_is_fitted(self, attributes=["embedding_", "columns_"])

        if column not in self.columns_:
            raise ValueError(f"Unknown fitted column: {column}")

        table = self.column_embedding_tables_[column]
        df = pd.DataFrame(
            table,
            columns=[f"{column}_pce_{i}" for i in range(self.n_components)],
        )
        df.insert(0, "canonical_value", self.vocab_[column].tolist())
        df.insert(1, "local_code", np.arange(len(df), dtype=np.int32))
        return df

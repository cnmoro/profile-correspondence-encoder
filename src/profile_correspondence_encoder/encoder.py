from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from time import perf_counter
from typing import Any
import unicodedata

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.validation import check_is_fitted


def _count_pair_compressed(
    global_a: np.ndarray,
    global_b: np.ndarray,
    total_nodes: int,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    if global_a.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    total_nodes_i64 = np.int64(total_nodes)

    if method == "unique":
        compressed = global_a.astype(np.int64) * total_nodes_i64 + global_b.astype(np.int64)
        uniq, cnt = np.unique(compressed, return_counts=True)
        return uniq.astype(np.int64, copy=False), cnt.astype(np.float64, copy=False)

    order = np.lexsort((global_b, global_a))
    a_sorted = global_a[order].astype(np.int64, copy=False)
    b_sorted = global_b[order].astype(np.int64, copy=False)

    change = np.empty(a_sorted.shape[0], dtype=bool)
    change[0] = True
    change[1:] = (a_sorted[1:] != a_sorted[:-1]) | (b_sorted[1:] != b_sorted[:-1])

    starts = np.flatnonzero(change)
    counts = np.diff(np.append(starts, a_sorted.shape[0]))

    uniq_a = a_sorted[starts]
    uniq_b = b_sorted[starts]
    uniq = uniq_a * total_nodes_i64 + uniq_b
    return uniq.astype(np.int64, copy=False), counts.astype(np.float64, copy=False)


class PCEEncoder(BaseEstimator, TransformerMixin):
    """Profile Correspondence Encoder (PCE).

    Scalable categorical encoder inspired by MCA/correspondence analysis.
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
        shrinkage_lambda: float = 5.0,
        pair_count_method: str = "lexsort",
        chunk_size: int | None = None,
        max_categories_per_column: int | None = None,
        pairing_strategy: str = "all",
        anchor_columns: list[str] | None = None,
        pair_sample_ratio: float = 1.0,
        n_jobs: int = 1,
        edge_weighting: str = "count",
        svd_algorithm: str = "randomized",
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
        self.shrinkage_lambda = float(shrinkage_lambda)
        self.pair_count_method = pair_count_method
        self.chunk_size = chunk_size
        self.max_categories_per_column = max_categories_per_column
        self.pairing_strategy = pairing_strategy
        self.anchor_columns = anchor_columns
        self.pair_sample_ratio = float(pair_sample_ratio)
        self.n_jobs = int(n_jobs)
        self.edge_weighting = edge_weighting
        self.svd_algorithm = svd_algorithm

    def _validate_params(self) -> None:
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")
        if self.min_frequency < 1:
            raise ValueError("min_frequency must be >= 1")
        if self.output_format not in {"numpy", "pandas"}:
            raise ValueError("output_format must be 'numpy' or 'pandas'")
        if self.pair_count_method not in {"lexsort", "unique"}:
            raise ValueError("pair_count_method must be 'lexsort' or 'unique'")
        if self.chunk_size is not None and int(self.chunk_size) < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.max_categories_per_column is not None and int(self.max_categories_per_column) < 1:
            raise ValueError("max_categories_per_column must be >= 1")
        if self.pairing_strategy not in {"all", "anchor", "sample"}:
            raise ValueError("pairing_strategy must be one of: all, anchor, sample")
        if not (0.0 < self.pair_sample_ratio <= 1.0):
            raise ValueError("pair_sample_ratio must be in (0, 1]")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        if self.edge_weighting not in {"count", "pmi", "ppmi"}:
            raise ValueError("edge_weighting must be one of: count, pmi, ppmi")
        if self.svd_algorithm not in {"randomized", "arpack"}:
            raise ValueError("svd_algorithm must be 'randomized' or 'arpack'")

    def _iter_row_slices(self, n_rows: int):
        if self.chunk_size is None or self.chunk_size >= n_rows:
            yield slice(0, n_rows)
            return
        step = int(self.chunk_size)
        for start in range(0, n_rows, step):
            yield slice(start, min(start + step, n_rows))

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
                    ch for ch in unicodedata.normalize("NFKD", str(x)) if not unicodedata.combining(ch)
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

    def _prepare_series(self, raw: pd.Series, col: str) -> pd.Series:
        return self._apply_aliases(self._normalize_series(raw), col)

    def _merge_counts(self, target: dict[str, int], counts: pd.Series) -> None:
        for k, v in counts.items():
            target[str(k)] = target.get(str(k), 0) + int(v)

    def _build_column_vocab(self, counts: pd.Series):
        frequent = counts[counts >= self.min_frequency]
        if self.max_categories_per_column is not None:
            frequent = frequent.nlargest(int(self.max_categories_per_column))
        kept = frequent.index.tolist()
        for token in (self.rare_token, self.unknown_token, self.missing_token):
            if token not in kept:
                kept.append(token)
        vocab = pd.Index(pd.unique(pd.Series(kept, dtype="string")))
        vocab_to_code = {v: i for i, v in enumerate(vocab.tolist())}
        return vocab, vocab_to_code

    def _resolve_pairs(self) -> list[tuple[str, str]]:
        all_pairs = list(combinations(self.columns_, 2))
        if self.pairing_strategy == "all":
            return all_pairs
        if self.pairing_strategy == "anchor":
            anchors = set(self.anchor_columns or [])
            if not anchors:
                raise ValueError("anchor_columns must be provided for pairing_strategy='anchor'")
            col_order = {c: i for i, c in enumerate(self.columns_)}
            pairs = set()
            for anchor in anchors:
                if anchor not in col_order:
                    raise ValueError(f"anchor column '{anchor}' is not in fitted columns")
                for other in self.columns_:
                    if other == anchor:
                        continue
                    a, b = (anchor, other) if col_order[anchor] < col_order[other] else (other, anchor)
                    pairs.add((a, b))
            return sorted(pairs, key=lambda p: (col_order[p[0]], col_order[p[1]]))
        if len(all_pairs) == 0:
            return all_pairs
        rng = np.random.default_rng(self.random_state)
        n_keep = max(1, int(np.ceil(len(all_pairs) * self.pair_sample_ratio)))
        idx = np.sort(rng.choice(len(all_pairs), size=n_keep, replace=False))
        return [all_pairs[i] for i in idx.tolist()]

    def _encode_series_for_fit_or_transform(self, s: pd.Series, col: str):
        counts = self.counts_[col]
        vocab_to_code = self.vocab_to_code_[col]
        cnt = s.map(counts).fillna(0).astype(np.int64).to_numpy()
        mapped = s.map(vocab_to_code).fillna(-1).astype(np.int32).to_numpy()
        in_vocab = mapped >= 0
        is_frequent_in_vocab = (cnt >= self.min_frequency) & in_vocab
        is_seen = cnt > 0
        rare_code = np.int32(vocab_to_code[self.rare_token])
        unk_code = np.int32(vocab_to_code[self.unknown_token])
        out = np.where(is_frequent_in_vocab, mapped, np.where(is_seen, rare_code, unk_code)).astype(np.int32)
        bucketed_seen = is_seen & ~is_frequent_in_vocab
        return out, cnt, bucketed_seen

    def _apply_edge_weighting(self, graph: sparse.csr_matrix) -> sparse.csr_matrix:
        if self.edge_weighting == "count":
            return graph
        coo = graph.tocoo(copy=True)
        data = coo.data.astype(np.float64, copy=False)
        total = float(data.sum())
        if total <= 0:
            return graph
        degree = np.asarray(graph.sum(axis=1)).ravel().astype(np.float64)
        eps = 1e-12
        denom = np.maximum(degree[coo.row] * degree[coo.col], eps)
        pmi = np.log(np.maximum((data * total) / denom, eps))
        if self.edge_weighting == "ppmi":
            pmi = np.maximum(pmi, 0.0)
        weighted = sparse.coo_matrix((pmi.astype(self.dtype, copy=False), (coo.row, coo.col)), shape=graph.shape)
        weighted = weighted.tocsr()
        weighted.eliminate_zeros()
        return weighted

    def _build_graph_streaming(self, X: pd.DataFrame) -> sparse.csr_matrix:
        edge_counts: dict[int, float] = defaultdict(float)
        t0 = perf_counter()
        for sl in self._iter_row_slices(len(X)):
            chunk = X.iloc[sl]
            local_codes: dict[str, np.ndarray] = {}
            for col in self.columns_:
                s = self._prepare_series(chunk[col], col)
                codes, _, _ = self._encode_series_for_fit_or_transform(s, col)
                local_codes[col] = codes

            pair_inputs = []
            for col_a, col_b in self.pairs_:
                global_a = local_codes[col_a].astype(np.int64) + np.int64(self.offsets_[col_a])
                global_b = local_codes[col_b].astype(np.int64) + np.int64(self.offsets_[col_b])
                pair_inputs.append((global_a, global_b, self.total_nodes_, self.pair_count_method))

            if self.n_jobs != 1 and len(pair_inputs) > 1:
                results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                    delayed(_count_pair_compressed)(*args) for args in pair_inputs
                )
            else:
                results = [_count_pair_compressed(*args) for args in pair_inputs]

            for uniq, cnt in results:
                for u, c in zip(uniq.tolist(), cnt.tolist()):
                    edge_counts[int(u)] += float(c)

        self.fit_stats_["pair_count_time_s"] = perf_counter() - t0

        if not edge_counts:
            raise ValueError("PCEEncoder requires at least 2 categorical columns with non-empty data.")

        keys = np.fromiter(edge_counts.keys(), dtype=np.int64)
        vals = np.fromiter(edge_counts.values(), dtype=np.float64).astype(self.dtype, copy=False)
        rows = (keys // np.int64(self.total_nodes_)).astype(np.int32)
        cols = (keys % np.int64(self.total_nodes_)).astype(np.int32)

        graph = sparse.coo_matrix((vals, (rows, cols)), shape=(self.total_nodes_, self.total_nodes_), dtype=self.dtype)
        graph = (graph + graph.T).tocsr()
        graph.sum_duplicates()
        graph.eliminate_zeros()
        return graph

    def _fit_embedding(self, graph: sparse.csr_matrix) -> np.ndarray:
        graph = self._apply_edge_weighting(graph)
        degree = np.asarray(graph.sum(axis=1)).ravel().astype(np.float64)
        degree = np.maximum(degree, 1.0)
        inv_sqrt_degree = 1.0 / np.sqrt(degree)
        d_inv_sqrt = sparse.diags(inv_sqrt_degree)
        normalized = d_inv_sqrt @ graph @ d_inv_sqrt

        k = min(self.n_components, max(1, normalized.shape[0] - 1))
        svd = TruncatedSVD(
            n_components=k,
            n_iter=self.svd_n_iter,
            random_state=self.random_state,
            algorithm=self.svd_algorithm,
        )
        embedding = svd.fit_transform(normalized).astype(self.dtype, copy=False)
        if k < self.n_components:
            pad = np.zeros((embedding.shape[0], self.n_components - k), dtype=self.dtype)
            embedding = np.hstack([embedding, pad])
        return embedding

    def fit(self, X, y=None):
        self._validate_params()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.fit_stats_ = {}
        fit_start = perf_counter()
        self.columns_ = self._resolve_columns(X)
        if len(self.columns_) < 2:
            raise ValueError("PCEEncoder requires at least 2 columns.")
        self.alias_maps_ = self._build_alias_maps()

        t0 = perf_counter()
        counts_maps: dict[str, dict[str, int]] = {col: {} for col in self.columns_}
        raw_maps: dict[str, dict[str, str]] = {col: {} for col in self.columns_}

        for sl in self._iter_row_slices(len(X)):
            chunk = X.iloc[sl]
            for col in self.columns_:
                s = self._prepare_series(chunk[col], col)
                self._merge_counts(counts_maps[col], s.value_counts(dropna=False))
                raw = chunk[col].astype("string")
                raw = raw.where(~raw.isna(), self.missing_token)
                m = pd.DataFrame({"raw": raw, "canon": s}).drop_duplicates()
                for r, c in zip(m["raw"].tolist(), m["canon"].tolist()):
                    raw_maps[col].setdefault(str(r), str(c))

        self.fit_stats_["pass1_time_s"] = perf_counter() - t0

        self.counts_ = {}
        self.vocab_ = {}
        self.vocab_to_code_ = {}
        self.local_code_to_value_ = {}
        self.raw_to_canonical_ = {}

        for col in self.columns_:
            counts = pd.Series(counts_maps[col], dtype="int64").sort_values(ascending=False)
            self.counts_[col] = counts
            vocab, vocab_to_code = self._build_column_vocab(counts)
            self.vocab_[col] = vocab
            self.vocab_to_code_[col] = vocab_to_code
            self.local_code_to_value_[col] = np.array(vocab.tolist(), dtype=object)
            self.raw_to_canonical_[col] = pd.DataFrame(
                {"raw_value": list(raw_maps[col].keys()), "canonical_value": list(raw_maps[col].values())}
            )

        self.offsets_ = {}
        self.total_nodes_ = 0
        for col in self.columns_:
            self.offsets_[col] = self.total_nodes_
            self.total_nodes_ += len(self.vocab_[col])

        self.pairs_ = self._resolve_pairs()
        graph = self._build_graph_streaming(X)

        t1 = perf_counter()
        self.embedding_ = self._fit_embedding(graph)
        self.fit_stats_["svd_time_s"] = perf_counter() - t1

        self.column_embedding_tables_ = {}
        self.column_default_vectors_ = {}
        self.column_rare_vectors_ = {}

        for col in self.columns_:
            start = self.offsets_[col]
            end = start + len(self.vocab_[col])
            table = self.embedding_[start:end]
            self.column_embedding_tables_[col] = table
            self.column_default_vectors_[col] = table[self.vocab_to_code_[col][self.unknown_token]]
            self.column_rare_vectors_[col] = table[self.vocab_to_code_[col][self.rare_token]]

        self.output_feature_names_ = [f"{col}_pce_{i}" for col in self.columns_ for i in range(self.n_components)]
        self.n_features_in_ = len(self.columns_)
        self.fit_stats_["fit_total_time_s"] = perf_counter() - fit_start
        return self

    def _transform_column_to_vectors(self, X_col: pd.Series, col: str) -> np.ndarray:
        s = self._prepare_series(X_col, col)
        codes, cnt, bucketed_seen = self._encode_series_for_fit_or_transform(s, col)
        table = self.column_embedding_tables_[col]
        codes = np.clip(codes, 0, table.shape[0] - 1)
        vectors = table[codes].astype(self.dtype, copy=True)

        if not self.enable_rare_shrinkage:
            return vectors

        specials = np.array([self.rare_token, self.unknown_token, self.missing_token], dtype=object)
        is_special = np.isin(s.to_numpy(dtype=object), specials)
        idx_mask = bucketed_seen & ~is_special
        if np.any(idx_mask):
            lam = max(self.shrinkage_lambda, 1e-12)
            alpha = (cnt.astype(np.float64) / (cnt.astype(np.float64) + lam)).clip(0.0, 1.0).astype(self.dtype)
            idx = np.where(idx_mask)[0]
            a = alpha[idx][:, None]
            rare_vec = self.column_rare_vectors_[col]
            unk_vec = self.column_default_vectors_[col]
            vectors[idx] = a * rare_vec + (1.0 - a) * unk_vec

        return vectors

    def transform(self, X):
        check_is_fitted(self, attributes=["embedding_", "columns_"])
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns_)

        n_rows = len(X)
        out = np.empty((n_rows, len(self.columns_) * self.n_components), dtype=self.dtype)
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
                        float(int(counts.get(v, 0)) / (int(counts.get(v, 0)) + max(self.shrinkage_lambda, 1e-12)))
                        if v not in {self.rare_token, self.unknown_token, self.missing_token}
                        else np.nan
                        for v in vocab.tolist()
                    ],
                }
            )
            raw_map_df = self.raw_to_canonical_[col].copy()
            raw_map_df["column"] = col
            frames.append(raw_map_df.merge(fitted_df, on=["column", "canonical_value"], how="left"))
        return pd.concat(frames, ignore_index=True)

    def get_column_embedding(self, column: str) -> pd.DataFrame:
        check_is_fitted(self, attributes=["embedding_", "columns_"])
        if column not in self.columns_:
            raise ValueError(f"Unknown fitted column: {column}")
        table = self.column_embedding_tables_[column]
        df = pd.DataFrame(table, columns=[f"{column}_pce_{i}" for i in range(self.n_components)])
        df.insert(0, "canonical_value", self.vocab_[column].tolist())
        df.insert(1, "local_code", np.arange(len(df), dtype=np.int32))
        return df

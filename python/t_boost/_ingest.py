"""Frame-ingestion helpers: polars-native input for the estimator layer.

polars is the first-class DataFrame type of the product layer (mirroring the rustystats GLM
library): eager ``pl.DataFrame`` and ``pl.LazyFrame`` are both accepted, per-row arguments
(``y``/``sample_weight``/``exposure``/``groups``) may reference frame columns by name, and
pandas is never involved. There is deliberately no import-time polars dependency: a polars
object can only reach this code if the caller already imported polars, so ``sys.modules``
sniffing is exact, free, and keeps the numpy-only native path dependency-clean.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

try:  # pandas is optional (never a dependency); used to detect pd.NA / pd.NaT
    from pandas import isna as _pd_isna
except ImportError:
    _pd_isna = None

# All missing markers in a categorical column collapse to this ONE reserved level, instead of
# splitting into distinct "nan" / "None" / "<NA>" strings (which every rival GBM avoids by routing
# missing to a single direction). Consistent between fit and serve.
_CAT_MISSING = "__tri_missing__"


def _cat_level(v: Any) -> str:
    """Stringify a categorical value, mapping any missing marker to the reserved `_CAT_MISSING`."""
    if v is None:
        return _CAT_MISSING
    if isinstance(v, float) and v != v:  # plain float NaN (np.float64 subclasses float too)
        return _CAT_MISSING
    if _pd_isna is not None:
        try:
            if bool(_pd_isna(v)):  # pd.NA / pd.NaT / np.datetime64('NaT')
                return _CAT_MISSING
        except (TypeError, ValueError):
            pass
    return str(v)


def _polars() -> Any:
    """The already-imported polars module, or ``None``.

    A polars frame/Series can only exist in-process if polars is in ``sys.modules``, so this
    never misses for real polars input and never triggers an import for everyone else.
    """
    return sys.modules.get("polars")


def is_polars_frame(x: Any) -> bool:
    """True for a polars ``DataFrame`` or ``LazyFrame``."""
    pl = _polars()
    return pl is not None and isinstance(x, (pl.DataFrame, pl.LazyFrame))


def is_polars_eager(x: Any) -> bool:
    """True for an eager polars ``DataFrame``."""
    pl = _polars()
    return pl is not None and isinstance(x, pl.DataFrame)


def collect_frame(x: Any, needed: list[str] | None = None) -> Any:
    """Collect a polars ``LazyFrame``; pass anything else through unchanged.

    ``needed`` restricts the collect to those columns (first-seen order, deduped) — the
    scan-level projection pushdown is what makes LazyFrame input worthwhile at serve time.
    ``None`` collects every column (the fit-time case: every non-consumed column is a
    feature, so nothing can be pruned).
    """
    pl = _polars()
    if pl is None or not isinstance(x, pl.LazyFrame):
        return x
    if needed is not None:
        try:
            return x.select(list(dict.fromkeys(needed))).collect()
        except pl.exceptions.ColumnNotFoundError as exc:
            raise ValueError(f"X (LazyFrame) is missing required column(s): {exc}") from exc
    return x.collect()


def select_feature_columns(x: Any, names: list[str]) -> Any:
    """Serve-time name-based selection on a polars frame: pick the model's feature columns
    in their fit-time order (extra columns are ignored, input column order is irrelevant);
    a LazyFrame collects only these columns. Missing columns raise ``ValueError``.
    """
    pl = _polars()
    try:
        if isinstance(x, pl.LazyFrame):
            return x.select(names).collect()
        # Eager: assemble the selected columns directly. `DataFrame.select` runs a query through
        # the lazy engine, a fixed ~0.3 ms per call — most of a small predict.
        return pl.DataFrame([x.get_column(n) for n in names])
    except pl.exceptions.ColumnNotFoundError as exc:
        raise ValueError(
            f"X is missing feature column(s) the model was fitted with: {exc}"
        ) from exc


def auto_categorical_idx(x: Any) -> list[int]:
    """Positional indices of the dtype-categorical columns of a polars ``DataFrame``.

    String / Categorical / Enum columns cannot be cast to a numeric design axis, so they are
    unambiguously categorical — no declaration needed. Returns ``[]`` for any other input
    type (numpy callers keep declaring categoricals explicitly).
    """
    pl = _polars()
    if pl is None or not isinstance(x, pl.DataFrame):
        return []
    return [
        i
        for i, dt in enumerate(x.dtypes)
        if dt == pl.String or isinstance(dt, (pl.Categorical, pl.Enum))
    ]


# Below this many rows, per-column Series work beats one parallel query through the lazy engine
# (whose fixed cost is ~0.3 ms a query); both produce identical arrays.
_POLARS_QUERY_MIN_ROWS = 16_384
# Below this many rows a categorical column is passed one label per row (see `_polars_cat_codes`).
_POLARS_CODES_MIN_ROWS = 1024


def _polars_cat_codes(s: Any, pl: Any) -> tuple[np.ndarray, list[str]]:
    """A polars categorical column as ``(codes, labels)``: row ``r``'s label is
    ``labels[codes[r]]``, exactly the label ``split_polars_columns`` gives it as a string list.

    String/Categorical/Enum values are their own labels, so the distinct non-null values become
    an ``Enum`` whose physical codes index them directly; nulls take a trailing
    ``_CAT_MISSING``. Numeric-dtype declared categoricals, and small columns, pass one label per
    row (codes ``0..n``), which is what the string-list path costs anyway.
    """
    n = s.len()
    dt = s.dtype
    if isinstance(dt, (pl.Categorical, pl.Enum)):
        s = s.cast(pl.String)
    elif dt != pl.String:
        return np.arange(n, dtype=np.uint32), [_cat_level(v) for v in s.to_list()]
    if n < _POLARS_CODES_MIN_ROWS:
        labels = [_CAT_MISSING if v is None else v for v in s.to_list()]
        return np.arange(n, dtype=np.uint32), labels
    uniq = s.drop_nulls().unique()
    labels = uniq.to_list() + [_CAT_MISSING]
    codes = s.cast(pl.Enum(uniq)).to_physical().fill_null(len(labels) - 1)
    return np.ascontiguousarray(codes.to_numpy(), dtype=np.uint32), labels


def split_polars_columns(
    df: Any, cat_idx: list[int], *, coded: bool = False
) -> tuple[np.ndarray, Any, bool]:
    """Split an eager polars ``DataFrame`` into the native core's design inputs.

    Returns ``(numeric_x, cat_x, needs_precision_warn)``: the numeric block as an
    F-contiguous float32 ndarray (polars' ``to_numpy`` default order — the native layer's
    one-memcpy-per-column ingest path, no transpose), the categorical columns as per-column
    string lists with nulls/NaN collapsed to ``_CAT_MISSING``, and whether any
    numeric-treated column was cast from a non-Float32 dtype (the caller owns the
    once-per-estimator ``PrecisionWarning``). Numeric nulls become NaN, which the core
    routes to the reserved missing bin.
    """
    pl = _polars()
    schema = df.schema
    names = df.columns
    cat_set = set(cat_idx)

    bad_cat = [
        f"{names[j]!r} ({schema[names[j]]})"
        for j in cat_idx
        if schema[names[j]].is_nested()
        or schema[names[j]] in (pl.Object, pl.Null, pl.Unknown)
    ]
    if bad_cat:
        raise ValueError(
            "categorical column(s) have dtypes with no well-defined levels: "
            + ", ".join(bad_cat)
            + ". Cast them to String (or a numeric dtype) explicitly."
        )

    numeric_names = [n for i, n in enumerate(names) if i not in cat_set]
    bad_numeric = [
        f"{n!r} ({schema[n]})"
        for n in numeric_names
        if not (schema[n].is_numeric() or schema[n] == pl.Boolean)
    ]
    if bad_numeric:
        raise ValueError(
            "polars column(s) treated as numeric have non-numeric dtypes: "
            + ", ".join(bad_numeric)
            + ". String/Categorical/Enum columns are categorical features automatically; "
            "anything else (temporal/nested/object) must be converted explicitly — cast it "
            "to a numeric dtype, or name it in categorical_features."
        )

    if numeric_names and df.height < _POLARS_QUERY_MIN_ROWS:
        # Same cast kernel and null -> NaN conversion as the query below, one column at a time.
        numeric_x = np.empty((df.height, len(numeric_names)), dtype=np.float32, order="F")
        for i, n in enumerate(numeric_names):
            numeric_x[:, i] = df.get_column(n).cast(pl.Float32).to_numpy()
    elif numeric_names:
        numeric_x = df.select(pl.col(n).cast(pl.Float32) for n in numeric_names).to_numpy()
    else:
        # select([]) would lose the row count; the core accepts a 0-column numeric block.
        numeric_x = np.empty((df.height, 0), dtype=np.float32, order="F")
    needs_warn = any(schema[n] != pl.Float32 for n in numeric_names)

    def cat_levels(name: str) -> list[str]:
        dt = schema[name]
        # String/Categorical/Enum values are str or null — `_cat_level` reduces to the
        # identity for non-nulls, so `fill_null` is level-identical at ~3.4x less wall
        # (measured: 0.31s -> 0.09s per MTPL-scale fit; the per-value Python loop was
        # 1.6M `_cat_level` calls). Categorical/Enum cast to String FIRST: an Enum
        # `fill_null` with a value outside its category set silently leaves the null in
        # place (no error), which would leak None into the core as a level. Only
        # numeric-dtype DECLARED categoricals need the per-value path (float NaN and
        # str() formatting have no polars equivalent).
        levels: list[str]
        if dt == pl.String:
            levels = df[name].fill_null(_CAT_MISSING).to_list()
        elif isinstance(dt, (pl.Categorical, pl.Enum)):
            levels = df[name].cast(pl.String).fill_null(_CAT_MISSING).to_list()
        else:
            levels = [_cat_level(v) for v in df[name].to_list()]
        return levels

    if coded:
        coded_x = [_polars_cat_codes(df.get_column(names[j]), pl) for j in cat_idx] or None
        return numeric_x, coded_x, needs_warn
    cat_x = [cat_levels(names[j]) for j in cat_idx] or None
    return numeric_x, cat_x, needs_warn


def resolve_vector(
    frame: Any, arg: Any, name: str, consumed: list[str] | None = None
) -> Any:
    """Resolve a per-row vector argument (``y``/``sample_weight``/``exposure``/``groups``).

    A ``str`` names a column of the polars frame ``X`` (rustystats-style); the column is
    returned as numpy and its name appended to ``consumed`` so the caller can drop it from
    the feature set. A polars ``Series`` converts to numpy. Anything else (numpy, lists,
    ``None``) passes through untouched. ``frame`` is the eager polars ``DataFrame`` X, or
    ``None`` when X is not polars — a column-name string then raises ``TypeError``.
    """
    if arg is None:
        return None
    pl = _polars()
    if isinstance(arg, str):
        if pl is None or frame is None or not isinstance(frame, pl.DataFrame):
            raise TypeError(
                f"{name}={arg!r} is a column-name string, which requires X to be a polars "
                "DataFrame or LazyFrame"
            )
        try:
            column = frame[arg]
        except pl.exceptions.ColumnNotFoundError as exc:
            raise ValueError(f"{name}={arg!r} is not a column of X: {exc}") from exc
        nulls = int(column.null_count())
        if nulls:
            # A null here would silently poison the fit as NaN (y/weight/exposure) or break
            # the group carve; unlike a FEATURE column, there is no missing-bin semantic.
            raise ValueError(
                f"{name}={arg!r} column contains {nulls} null(s); "
                "fill or drop those rows before fitting"
            )
        if consumed is not None:
            consumed.append(arg)
        return column.to_numpy()
    if pl is not None and isinstance(arg, pl.Series):
        return arg.to_numpy()
    return arg

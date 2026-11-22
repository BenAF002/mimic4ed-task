"""Polars helpers for general processing of the master dataset.

Outlier handling, imputation, train/test splitting, clinical-score computation and ED vital-sign
resampling. The thresholds and score definitions live in ``config.json``, not here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

CONFIG_PATH = Path(__file__).parent / "config.json"
FrameT = pl.DataFrame | pl.LazyFrame

_BOUND_OPS = {
    "gt": lambda c, v: c > v,
    "ge": lambda c, v: c >= v,
    "lt": lambda c, v: c < v,
    "le": lambda c, v: c <= v,
}


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    """Load the processing config (vital-sign ranges, derived columns and score definitions)."""
    with open(path) as f:
        return json.load(f)


# ---- vital signs: temperature, outliers, imputation --------------------------------
def vital_type(column: str) -> str | None:
    """Vital-sign type of a column: its 2nd underscore token (`triage_sbp` -> `sbp`), if any."""
    parts = column.split("_")
    return parts[1] if len(parts) > 1 else None


def vital_columns(columns: list[str], ranges: dict, exclude_types: tuple[str, ...] = ()) -> list[str]:
    """Columns whose vital type is in `ranges` and not in `exclude_types`."""
    return [c for c in columns if vital_type(c) in ranges and vital_type(c) not in exclude_types]


def convert_temp_to_celsius(df: FrameT) -> FrameT:
    """Convert every temperature column from Fahrenheit to Celsius."""
    cols = [c for c in df.collect_schema().names() if vital_type(c) == "temperature"]
    return df.with_columns((pl.col(c).cast(pl.Float64) - 32) * (5 / 9) for c in cols)


def _outlier_expr(col: str, r: dict) -> pl.Expr:
    x = pl.col(col).cast(pl.Float64).fill_nan(None)
    return (
        pl.when((x < r["outlier_low"]) | (x > r["outlier_high"])).then(None)  # outlier -> missing
        .when(x < r["valid_low"]).then(pl.lit(r["valid_low"], dtype=pl.Float64))  # clip to valid bounds
        .when(x > r["valid_high"]).then(pl.lit(r["valid_high"], dtype=pl.Float64))
        .otherwise(x)
        .alias(col)
    )


def remove_outliers(df: FrameT, ranges: dict) -> FrameT:
    """Null out values beyond the outlier bounds and clip values beyond the valid bounds."""
    cols = vital_columns(df.collect_schema().names(), ranges)
    return df.with_columns(_outlier_expr(c, ranges[vital_type(c)]) for c in cols)


def outlier_counts(df: FrameT, ranges: dict) -> pl.DataFrame:
    """Per vital-sign column, the number of values in each outlier / valid band."""
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    rows = []
    for c in vital_columns(df.columns, ranges):
        r, x = ranges[vital_type(c)], pl.col(c)
        rows.append(df.select(
            pl.lit(c).alias("variable"),
            (x < r["outlier_low"]).sum().alias("< outlier_low"),
            ((x >= r["outlier_low"]) & (x < r["valid_low"])).sum().alias("[outlier_low, valid_low)"),
            ((x >= r["valid_low"]) & (x <= r["valid_high"])).sum().alias("[valid_low, valid_high]"),
            ((x > r["valid_high"]) & (x <= r["outlier_high"])).sum().alias("(valid_high, outlier_high]"),
            (x > r["outlier_high"]).sum().alias("> outlier_high"),
        ))
    return pl.concat(rows)


def impute_median(train: pl.DataFrame, others: list[pl.DataFrame], columns: list[str]
                  ) -> tuple[pl.DataFrame, list[pl.DataFrame], dict[str, float]]:
    """Fill nulls in `columns` with the `train` median; returns (train, others, medians)."""
    medians = train.select(pl.col(columns).median()).row(0, named=True)
    fill = [pl.col(c).fill_null(m) for c, m in medians.items()]
    return train.with_columns(fill), [df.with_columns(fill) for df in others], medians


# ---- train / test split ----------------------------------------------------------------
def train_test_split(df: pl.DataFrame, frac: float = 0.8, seed: int = 10
                     ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split like ``pandas.DataFrame.sample(frac, random_state=seed)``.

    Train rows come out in sampled order; test is the remainder in original row order.
    """
    n = df.height
    train_idx = np.random.RandomState(seed).choice(n, size=round(frac * n), replace=False)
    is_test = np.ones(n, dtype=bool)
    is_test[train_idx] = False
    return df[train_idx], df.filter(pl.Series(is_test))


# ---- scores ----------------------------------------------------------------------------
def add_derived(df: FrameT, derived: dict) -> FrameT:
    """Add derived columns defined as sums of ``column * mul / div`` terms (e.g. MAP)."""
    return df.with_columns(
        sum((pl.col(t["column"]) * t["mul"] / t["div"] for t in terms[1:]),
            pl.col(terms[0]["column"]) * terms[0]["mul"] / terms[0]["div"]).alias(name)
        for name, terms in derived.items()
    )


def _bin_condition(column: str, b: dict) -> pl.Expr:
    conds = [_BOUND_OPS[k](pl.col(column), b[k]) for k in _BOUND_OPS if k in b]
    cond = conds[0]
    for c in conds[1:]:
        cond = cond & c
    return cond


def score_expr(components: list[dict]) -> pl.Expr:
    """Expression summing a score's components (see ``config.json`` for the format)."""
    terms = []
    for comp in components:
        if "bins" in comp:  # first matching bin wins, else 0 (like numpy.select)
            expr = pl.when(_bin_condition(comp["column"], comp["bins"][0])).then(comp["bins"][0]["points"])
            for b in comp["bins"][1:]:
                expr = expr.when(_bin_condition(comp["column"], b)).then(b["points"])
            terms.append(expr.otherwise(0).cast(pl.Int64))
        else:
            terms.append(pl.sum_horizontal(pl.col(c).cast(pl.Int64) * w for c, w in comp["weights"].items()))
    return pl.sum_horizontal(terms)


def add_scores(df: FrameT, config: dict, names: list[str]) -> FrameT:
    """Add derived columns, then ``score_<name>`` for each requested score."""
    df = add_derived(df, config["derived"])
    return df.with_columns(score_expr(config["scores"][n]).alias(f"score_{n}") for n in names)


# ---- ED vital-sign time series -------------------------------------------------------------
def resample_vitals(vitals: pl.LazyFrame, value_cols: list[str], freq: str = "1h") -> pl.LazyFrame:
    """Resample each stay's vital signs onto a regular grid starting at its first chart time.

    Observations are floored to whole minutes from the first chart time, linearly interpolated in
    time, and sampled on the grid; the last value is held after the final observation and the first
    grid value is back-filled into any leading gap. Matches the legacy pandas
    ``resample('1T', origin='start').interpolate()`` followed by an ``asfreq`` at `freq`, except that
    stays with sub-minute chart times (~0.001% of stays) can differ slightly in interpolated values.
    """
    bounds = vitals.group_by("subject_id", "stay_id").agg(
        _start=pl.col("charttime").min(), _end=pl.col("charttime").max()
    )
    obs = (  # floor each observation to its minute bin (anchored at the stay's first chart time)
        vitals.join(bounds, on=["subject_id", "stay_id"])
        .sort("charttime")
        .with_columns(charttime=pl.col("_start") + pl.duration(minutes=(pl.col("charttime") - pl.col("_start")).dt.total_minutes()))
        .group_by("subject_id", "stay_id", "charttime", maintain_order=True)
        .agg(pl.col(c).cast(pl.Float64).fill_nan(None).drop_nulls().last() for c in value_cols)
    )
    grid = (
        bounds.select("subject_id", "stay_id",
                      charttime=pl.datetime_ranges("_start", "_end", interval=freq, time_unit="us"))
        .explode("charttime")
        .with_columns(_grid=pl.lit(True))
    )
    obs = obs.join(grid.select("stay_id", "charttime", "_grid"), on=["stay_id", "charttime"], how="left")
    extra = grid.join(obs.select("stay_id", "charttime"), on=["stay_id", "charttime"], how="anti")
    combined = pl.concat([
        obs.select("subject_id", "stay_id", "charttime", *[pl.col(c).cast(pl.Float64) for c in value_cols],
                   _grid=pl.col("_grid").fill_null(False)),
        extra.select("subject_id", "stay_id", "charttime",
                     *[pl.lit(None, dtype=pl.Float64).alias(c) for c in value_cols], "_grid"),
    ]).sort("stay_id", "charttime")
    return (
        combined
        .with_columns(pl.col(c).cast(pl.Float64).fill_nan(None).interpolate_by("charttime").over("stay_id") for c in value_cols)
        .with_columns(pl.col(c).forward_fill().over("stay_id") for c in value_cols)  # hold last value after final obs
        .filter("_grid")
        .with_columns(pl.col(c).backward_fill().over("stay_id") for c in value_cols)  # leading gaps
        .drop("_grid")
    )

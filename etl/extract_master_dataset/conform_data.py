import polars as pl
import numpy as np
import re
import collections


def conform_dates(df, date_cols: list[str]):
    """Conform cols in ``date_cols`` to datetime format."""
    # Columns can mix formats (e.g. ISO and M/D/YYYY), so try each in turn.
    formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y"]
    schema = df.collect_schema()
    df = df.with_columns([
        pl.coalesce([
            pl.col(col).str.to_datetime(fmt, strict=False) for fmt in formats
        ]).alias(col)
        for col in date_cols
        if schema[col] == pl.String
    ])
    return df

# copied from original `helpers.py` module
def convert_str_to_float(x):
    if isinstance(x, str):
        x_split = re.compile('[^a-zA-Z0-9-]').split(x.strip())
        if '-' in x_split[0]:
            x_split_dash = x_split[0].split('-')
            if len(x_split_dash) == 2 and x_split_dash[0].isnumeric() and x_split_dash[1].isnumeric():
                return (float(x_split_dash[0]) + float(x_split_dash[1])) / 2
            else:
                return np.nan
        else:
            if x_split[0].isnumeric():
                return float(x_split[0])
            else:
                return np.nan
    else:
        return x


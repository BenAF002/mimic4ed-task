"""Polars versions of the diagnosis-history features: ICD lists with integer encoding and
Charlson / Elixhauser comorbidity flags.

The legacy per-code helpers (charlson, elixhauser, convert_9to10_list, convert_10to9_list) are
called once per DISTINCT (icd_code, icd_version), not once per row, and the results are joined back.
This requires the helpers to be pure per-code functions.
"""
from pathlib import Path
import polars as pl
import json

# import legacy dependencies from medcodes
from medcodes import charlson, elixhauser
from medcodes.diagnoses.icd_conversion import convert_9to10_list, convert_10to9_list

# read the mappings.json file into map_dict and overwrite keys (matching original style)
with open(Path(__file__).parent / "mappings.json") as f:
    MAP_DICT = json.load(f)
MAP_DICT["charlson"] = MAP_DICT.pop("cci_vars")
MAP_DICT["elixhauser"] = MAP_DICT.pop("eci_vars")

VERSIONS = {'v9_3digit': (9, True), 'v9': (9, False), 'v10': (10, False)}

def prior_diagnoses(df_master, df_diagnoses, df_admissions, days: int) -> pl.DataFrame:
    """Long table (stay_id, icd_code, icd_version): diagnoses from admissions
    discharged in [intime - days, intime) for the same subject."""
    dx = (
        df_diagnoses
        .join(df_admissions.select('subject_id', 'hadm_id', 'dischtime'),
              on=['subject_id', 'hadm_id'], how='left')
        .select('subject_id', 'icd_code', 'icd_version', 'dischtime')
    )
    return (
        df_master
        .select('stay_id', 'subject_id', 'intime')
        .join(dx, on='subject_id', how='inner')
        .filter(
            (pl.col('dischtime') < pl.col('intime'))
            & (pl.col('intime') - pl.col('dischtime') <= pl.duration(days=days))
        )
        .select('stay_id', 'icd_code', 'icd_version')
    )


# ---- comorbidity flags -------------------------------------------------------
def build_comorbidity_lookup(codes: pl.DataFrame | pl.LazyFrame, charlson, elixhauser, 
                             map_dict: dict | None = None) -> pl.DataFrame:
    """Call the legacy per-code helpers once per DISTINCT (icd_code, icd_version)."""
    map_dict = MAP_DICT if map_dict is None else map_dict
    if isinstance(codes, pl.LazyFrame):
        codes = codes.collect()  # only the distinct codes get materialized
    rows = [
        (c, v, map_dict[scheme][cat])
        for c, v in codes.iter_rows()
        for scheme, fn in (('charlson', charlson), ('elixhauser', elixhauser))
        for cat in fn(c, v)
    ]
    return pl.DataFrame(
        rows, orient='row',
        schema={'icd_code': pl.String, 'icd_version': pl.Int64, 'category': pl.String}
    )


def comorbidity(df_master, df_diagnoses, df_admissions, 
                 days: int, map_dict: dict | None = None) -> pl.DataFrame:
    """Build comorbidity flags for each stay based on prior diagnoses."""
    map_dict = MAP_DICT if map_dict is None else map_dict
    empty_map_vector = {k:{v:0 for _, v in map_i.items()} for k, map_i in map_dict.items()}
    categories = [c for s in ('charlson', 'elixhauser') for c in empty_map_vector[s]]
    
    if len(set(categories)) != len(categories):
        raise ValueError("""Charlson and Elixhauser share column names. 
                            The legacy dict merge would let Elixhauser silently overwrite Charlson.""")
    
    prior = prior_diagnoses(df_master, df_diagnoses, df_admissions, days)
    lookup = build_comorbidity_lookup(prior.select('icd_code', 'icd_version').unique(), 
                                      charlson, elixhauser, map_dict)
    if isinstance(prior, pl.LazyFrame):
        lookup = lookup.lazy()
    flags = (
        prior.join(lookup, on=['icd_code', 'icd_version'])
        .select('stay_id', 'category').unique()
        .group_by('stay_id')
        .agg((pl.col('category') == c).any().cast(pl.Int8).alias(c) for c in categories)
    )
    return (
        df_master.join(flags, on='stay_id', how='left', maintain_order='left')
        .with_columns(pl.col(c).fill_null(0) for c in categories)
    )


# ---- ICD lists + integer encoding -------------------------------------------------
def icd_list(df_edstays, df_diagnoses, df_admissions, days: int, 
             version: int = 9, digit3: bool = False):
    """Generate ICD code lists and their integer encodings for each stay."""
    prior = prior_diagnoses(df_edstays, df_diagnoses, df_admissions, days)
    codes = prior.select('icd_code', 'icd_version').unique()
    if version == 10:
        convert = (lambda c, v: convert_9to10_list([c], [v]))
    else:
        convert = (lambda c, v: convert_10to9_list([c], [v], digit3=digit3))
    if isinstance(codes, pl.LazyFrame):
        codes = codes.collect()
    xwalk = pl.DataFrame(
        [(c, v, t) for c, v in codes.iter_rows()
         for t in convert(c, v)],
        orient='row', schema={'icd_code': pl.String, 'icd_version': pl.Int64, 'code': pl.String})

    if isinstance(prior, pl.LazyFrame):
        xwalk = xwalk.lazy()
    long = prior.join(xwalk, on=['icd_code', 'icd_version']).select('stay_id', 'code').unique()
    encode = (
        long.select('code').unique().sort('code')
        .with_row_index('idx')
        .with_columns(pl.col('idx').cast(pl.Int64))
    )

    per_stay = (
        long.join(encode, on='code').sort('stay_id', 'idx')
        .group_by('stay_id', maintain_order=True)
        .agg(icd_list=pl.col('code'), icd_encoded_list=pl.col('idx'))
    )
    out = (
        df_edstays.select('stay_id')
        .join(per_stay, on='stay_id', how='left', maintain_order='left')
        .with_columns(
            pl.col('icd_list').fill_null(pl.lit([], dtype=pl.List(pl.String))),
            pl.col('icd_encoded_list').fill_null(pl.lit([], dtype=pl.List(pl.Int64))),
        )
    )
    return out, encode


def extract_icd_list(df_edstays, df_diagnoses, df_admissions, output_path: str | Path,
                     days: int = 365 * 5, version: str = 'v9_3digit'):
    """Write the per-stay ICD list and the code -> index encoding map to parquet files."""
    target, digit3 = VERSIONS[version]
    df, encode = icd_list(df_edstays, df_diagnoses, df_admissions, days, target, digit3)
    if isinstance(df, pl.LazyFrame):
        df, encode = pl.collect_all([df, encode])
    out = Path(output_path)
    df.write_parquet(out / f'icd_list_dataset_{version}.parquet')      # list columns cannot go to CSV
    encode.write_parquet(out / f'icd_encode_map_{version}.parquet')    # columns: idx, code
    print(f"ICD List written to:     {out / f'icd_list_dataset_{version}.parquet'}")
    print(f"Encoding map written to: {out / f'icd_encode_map_{version}.parquet'}")
    print(f'Number of unique ICD codes {version}: {encode.height}')
    return df, encode

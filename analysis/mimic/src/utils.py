from __future__ import annotations

"""
Compatibility utilities for the public reproducibility repository.

The original local `src/utils.py` was not included in the exported upstream
candidate bundle. This module reconstructs only non-scientific infrastructure
used by the frozen scripts: DuckDB connection setup, source-file discovery,
DuckDB source expressions, SQL path quoting, and percentage formatting.

No cohort definition, exposure definition, censoring rule, weighting rule,
outcome definition, or causal-estimation logic is implemented here.
"""

from pathlib import Path
import duckdb


def sql_quote_path(path) -> str:
    return str(Path(path)).replace("'", "''")


def source_sql(path) -> str:
    p = Path(path)
    q = sql_quote_path(p)
    name = p.name.lower()
    if name.endswith(".parquet"):
        return f"read_parquet('{q}')"
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return f"read_csv_auto('{q}', header=true, sample_size=-1)"
    raise ValueError(f"Unsupported tabular source: {p}")


def try_find_table(data_root, module: str, table: str):
    root = Path(data_root)
    candidates = [
        root / module / f"{table}.csv.gz",
        root / module / f"{table}.csv",
        root / module / f"{table}.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Tolerate one additional release-directory layer or similarly organized
    # authorized local copies. Prefer exact module/table paths when possible.
    suffixes = [
        Path(module) / f"{table}.csv.gz",
        Path(module) / f"{table}.csv",
        Path(module) / f"{table}.parquet",
    ]
    hits = []
    if root.exists():
        for suffix in suffixes:
            hits.extend(root.glob(f"**/{suffix.as_posix()}"))
    hits = [p for p in hits if p.is_file()]
    if hits:
        return sorted(hits, key=lambda p: (len(p.parts), str(p).lower()))[0]
    return None


def find_table(data_root, module: str, table: str):
    p = try_find_table(data_root, module, table)
    if p is None:
        raise FileNotFoundError(
            f"Could not locate {module}/{table} as .csv.gz, .csv, or .parquet "
            f"below {Path(data_root)}"
        )
    return p


def connect_duckdb(project_root, threads: int = 4):
    project_root = Path(project_root)
    cache_dir = project_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / "feasibility.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(f"PRAGMA threads={max(1, int(threads))}")
    return con


def pct(numerator, denominator):
    denominator = float(denominator)
    if denominator == 0:
        return float("nan")
    return 100.0 * float(numerator) / denominator

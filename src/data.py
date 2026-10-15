"""Paths and loaders for the competition data."""
import os
from pathlib import Path

import polars as pl

KAGGLE_INPUT = Path("/kaggle/input")
if KAGGLE_INPUT.exists():  # running inside a Kaggle notebook: locate the competition files wherever they are mounted
    ROOT = Path("/kaggle/working")
    hits = sorted(KAGGLE_INPUT.rglob("combine_results.csv"))
    if not hits:
        raise FileNotFoundError("Competition data not found under /kaggle/input - add the NFL Big Data Bowl 2027 input.")
    RAW = hits[0].parent
else:
    ROOT = Path(__file__).resolve().parents[1]
    RAW = ROOT / "data" / "cache" / "competitions" / "nfl-big-data-bowl-2027" / "nfl-big-data-bowl-2027"
if os.environ.get("BDB_RAW"):  # local test run of the notebook
    RAW, ROOT = Path(os.environ["BDB_RAW"]), Path.cwd()
PROCESSED = ROOT / "data" / "processed"
NA = ["NA"]


def read(name: str, **kw) -> pl.DataFrame:
    return pl.read_csv(RAW / f"{name}.csv", null_values=NA, infer_schema_length=100_000, **kw)


def players() -> pl.DataFrame:
    p = read("players").with_columns(
        pl.col("draft_overall_pick").cast(pl.Int64, strict=False),
        pl.col("draft_round").cast(pl.Int64, strict=False),
    )
    return p.with_columns(undrafted=pl.col("draft_overall_pick").is_null())


def combine_results() -> pl.DataFrame:
    num = ["ten_yd_split", "forty", "vertical", "broad_jump", "three_cone",
           "short_shuttle", "bench_reps", "ngs_college_production_score", "ngs_final_score"]
    return read("combine_results").with_columns(pl.col(num).cast(pl.Float64, strict=False))


def combine_tracking(drill_type: str | None = None) -> pl.DataFrame:
    df = read("combine_tracking").filter(pl.col("entity_type") == "PLAYER")
    if drill_type:
        df = df.filter(pl.col("drill_type") == drill_type)
    return df.with_columns(pl.col("time").str.to_datetime()).sort("nfl_id", "drill_name", "attempt", "time")


def reg_season_plays() -> pl.DataFrame:
    games = read("games").select("game_id", "season", "season_type", "week")
    return read("player_play").join(games, on="game_id").filter(pl.col("season_type") == "REG")

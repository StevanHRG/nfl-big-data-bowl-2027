"""Exploração dos drills de DL: estrutura das tentativas e perfil de arrancada."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import polars as pl
import data

ct = data.combine_tracking("SKILL_DRILLS_DL")
att = (ct.group_by("nfl_id", "drill_name", "attempt", "event_id")
       .agg(frames=pl.len(), dur=(pl.col("time").max() - pl.col("time").min()).dt.total_milliseconds() / 1000,
            s0=pl.col("s").first(), smax=pl.col("s").max(), amax=pl.col("a").max(),
            dist=pl.col("dis").sum(),
            dx=pl.col("x").last() - pl.col("x").first(), dy=pl.col("y").last() - pl.col("y").first()))
print("Attempts per drill:")
print(att.group_by("drill_name").agg(
    n=pl.len(), players=pl.col("nfl_id").n_unique(),
    att_per_player=pl.len() / pl.col("nfl_id").n_unique(),
    dur_med=pl.col("dur").median(), s0_med=pl.col("s0").median(),
    smax_med=pl.col("smax").median(), amax_med=pl.col("amax").median(), dist_med=pl.col("dist").median(),
).sort("players", descending=True))

# time gaps: is it a clean 10 Hz series?
gaps = ct.with_columns(dt=pl.col("time").diff().over("event_id", "nfl_id", "attempt").dt.total_milliseconds())
print("dt distribution (ms):", gaps["dt"].drop_nulls().value_counts().sort("count", descending=True).head(5).rows())

# one example of the front-start drill and the pass rush drill
for drill in ["FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", "PASS_RUSH_DRILL"]:
    ex = ct.filter(pl.col("drill_name") == drill)
    first = ex.select("nfl_id", "attempt").row(0)
    one = ex.filter((pl.col("nfl_id") == first[0]) & (pl.col("attempt") == first[1]))
    print(f"\n{drill} example nfl_id={first[0]} attempt={first[1]}")
    with pl.Config(tbl_rows=60, tbl_width_chars=200):
        print(one.select("time", "x", "y", "s", "a", "dis", "dir").head(40))

# 40-yard dash coverage for DL vs stopwatch
cr = data.combine_results().filter(pl.col("combine_position") == "DL")
f40 = data.combine_tracking("FORTY_YARD_DASH").select("nfl_id").unique()
print("\nDL with stopwatch 10yd split:", cr["ten_yd_split"].is_not_null().sum(), "/", len(cr))
print("DL with 40 tracking:", cr.join(f40, on="nfl_id").height)
print("DL with any DL drill tracking:", cr.join(ct.select("nfl_id").unique(), on="nfl_id").height)

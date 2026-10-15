"""Combine Bend Score (CBS): final metric, effect sizes in football units and the player table."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
import data
from features import centripetal

ARC = ["FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", "RUN_THE_HOOP_DRILL", "RUN_AND_CLUB_DRILL"]
PROC = data.PROCESSED
PROC.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2027)

cr = data.combine_results()
dl_ids = cr.filter(pl.col("combine_position") == "DL").select("nfl_id")
plr = data.players()

# --- Combine metric (every DL with tracking, not only those who played) ---
ct = data.combine_tracking("SKILL_DRILLS_DL").join(dl_ids, on="nfl_id").filter(pl.col("drill_name").is_in(ARC))
a = centripetal(ct, ["nfl_id", "drill_name", "attempt"], smooth=3)
a.write_parquet(PROC / "dl_combine_attempts.parquet")
best = a.group_by("nfl_id", "drill_name").agg(pl.col("ac_max").max())
print("peak ac by drill (yd/s²):", best.group_by("drill_name").agg(
    pl.col("ac_max").mean().round(2).alias("mean"), pl.col("ac_max").std().round(2).alias("sd")).rows())
w = best.pivot(on="drill_name", index="nfl_id", values="ac_max")
w = w.with_columns([((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(f"z_{c}") for c in ARC])
w = w.with_columns(n_arc=pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int8) for c in ARC]),
                   cbs_z=pl.mean_horizontal([f"z_{c}" for c in ARC]))
# friendly scale: 50 = average, 10 = 1 SD
w = w.with_columns(cbs=50 + 10 * (pl.col("cbs_z") - pl.col("cbs_z").mean()) / pl.col("cbs_z").std())
print(f"DL with a CBS: {w.filter(pl.col('cbs').is_not_null()).height} of {dl_ids.height}")

# --- in-game outcome ---
reg = data.reg_season_plays().join(dl_ids, on="nfl_id")
rush = reg.filter(pl.col("player_get_off").is_not_null()).with_columns(
    edge=pl.col("lined_up_position").is_in(["EDGE", "OLB"]).cast(pl.Int8),
    pressed=pl.col("time_to_pressure").is_not_null().cast(pl.Int8),
    sacked=pl.col("sack").fill_null(0))
rate_by_role = rush.group_by("lined_up_position").agg(pl.col("pressed").mean().alias("role_rate"))
rush = rush.join(rate_by_role, on="lined_up_position")
K = 100
g = (rush.group_by("nfl_id").agg(snaps=pl.len(), pressures=pl.col("pressed").sum(), sacks=pl.col("sacked").sum(),
                                 edge_share=pl.col("edge").mean(), exp_press=pl.col("role_rate").sum(),
                                 seasons=pl.col("season").n_unique())
     .with_columns(press_rate=pl.col("pressures") / pl.col("snaps"),
                   # pressures over expected for the alignment, per 100 rushes, shrunk
                   poe100=100 * (pl.col("pressures") - pl.col("exp_press")) / (pl.col("snaps") + K)))

tab = (plr.join(cr.select("nfl_id", "combine_weight", "ten_yd_split", "forty"), on="nfl_id")
       .join(dl_ids, on="nfl_id").join(w.select("nfl_id", "cbs", "cbs_z", "n_arc", *ARC), on="nfl_id", how="left")
       .join(g, on="nfl_id", how="left")
       .with_columns(log_pick=pl.col("draft_overall_pick").fill_null(260).log()))
tab.write_parquet(PROC / "dl_cbs_table.parquet")

an = tab.filter(pl.col("snaps") >= 50, pl.col("cbs").is_not_null()).to_pandas()
print(f"\nAnalysis: n={len(an)} (undrafted: {an['undrafted'].sum()})")

# --- effect size in football units ---
m = smf.wls("poe100 ~ cbs + log_pick + combine_weight + edge_share", an, weights=an["snaps"]).fit()
b, lo, hi = m.params["cbs"], *m.conf_int().loc["cbs"]
print(f"+10 CBS points (1 SD) -> {10*b:+.2f} pressures over expected per 100 rushes  95% CI [{10*lo:+.2f}, {10*hi:+.2f}]  p={m.pvalues['cbs']:.4f}")
lg = rush["pressed"].mean()
print(f"baseline pressure rate: {100*lg:.1f} per 100 rushes -> relative effect {10*b/(100*lg):+.0%}")
m2 = smf.wls("poe100 ~ cbs + log_pick + combine_weight + edge_share + ten_yd_split", an.dropna(subset=['ten_yd_split']),
             weights=an.dropna(subset=['ten_yd_split'])["snaps"]).fit()
print(f"  + 10yd split: {10*m2.params['cbs']:+.2f}  p={m2.pvalues['cbs']:.4f}  (n={int(m2.nobs)})")
# comparison: what draft slot is worth in the same model
print(f"  comparison: pick 10 vs pick 100 -> {m.params['log_pick']*(np.log(10)-np.log(100)):+.2f} per 100 rushes")

# quartiles (pooled, alignment-adjusted)
an["q"] = np.digitize(an["cbs"], np.percentile(an["cbs"], [25, 50, 75])) + 1
qt = an.groupby("q").apply(lambda d: 100 * (d["pressures"].sum() - d["exp_press"].sum()) / d["snaps"].sum(),
                           include_groups=False)
print("pressures over expected per 100 rushes, by CBS quartile:", qt.round(2).to_dict())
boot = []
for _ in range(2000):
    s = an.sample(len(an), replace=True, random_state=int(rng.integers(1e9)))
    f = lambda d: 100 * (d["pressures"].sum() - d["exp_press"].sum()) / d["snaps"].sum()
    boot.append(f(s[s["q"] == 4]) - f(s[s["q"] == 1]))
print(f"Q4 - Q1: {qt[4]-qt[1]:+.2f}  95% bootstrap CI [{np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}]")

# --- hidden talent: high CBS and late pick / undrafted ---
hid = (tab.filter(pl.col("cbs").is_not_null())
       .filter((pl.col("cbs") >= 60) & ((pl.col("draft_overall_pick") > 100) | pl.col("undrafted")))
       .sort("cbs", descending=True)
       .select("display_name", "nfl_position", "draft_year", "draft_overall_pick", "cbs", "snaps", "pressures", "poe100"))
print("\nCBS >= 60 with pick > 100 or undrafted:")
with pl.Config(tbl_rows=30, tbl_width_chars=200):
    print(hid)
print("\nTop 10 CBS (all):")
with pl.Config(tbl_rows=10, tbl_width_chars=200):
    print(tab.filter(pl.col("cbs").is_not_null()).sort("cbs", descending=True)
          .select("display_name", "nfl_position", "draft_overall_pick", "cbs", "snaps", "pressures", "poe100").head(10))

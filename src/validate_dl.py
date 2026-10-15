"""Turn-force (DL) validation:
1. robustness to drill choice (incl. only the drills that did NOT stand out in the screen)
2. not just speed (control for max speed; smoothing / percentile variants)
3. the same measure in games: Combine -> game transfer and link to pressure
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
import data
from features import centripetal

PROC = data.PROCESSED
PROC.mkdir(parents=True, exist_ok=True)
CORE = ["FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", "RUN_THE_HOOP_DRILL", "RUN_AND_CLUB_DRILL",
        "FOUR_BAG_AGILITY_DRILL", "PASS_RUSH_DRILL", "BODY_CONTROL_DRILL"]
SCREEN_TOP = CORE[:3]   # the 3 drills at the top of the initial screen
BLIND = CORE[3:]        # the 3 drills that did NOT stand out


def zcols(df, cols):
    return df.with_columns([((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c) for c in cols])


def fit(d, outcome, base, x, label):
    d = d.select(outcome, x, *base).drop_nulls().to_pandas()
    z = (d - d.mean()) / d.std()
    m = smf.ols(f"{outcome} ~ {' + '.join(base + [x])}", z).fit()
    print(f"  {label:58s} n={len(d):3d} beta={m.params[x]:+.2f} p={m.pvalues[x]:.4f}")
    return m


cr = data.combine_results()
dl_ids = cr.filter(pl.col("combine_position") == "DL").select("nfl_id")
plr = data.players().select("nfl_id", "draft_overall_pick", "draft_year", "display_name", "nfl_position")

# ---------- Combine ----------
ct = data.combine_tracking("SKILL_DRILLS_DL").join(dl_ids, on="nfl_id").filter(pl.col("drill_name").is_in(CORE))
variants = {}
for sm in (1, 3, 5):
    a = centripetal(ct, ["nfl_id", "drill_name", "attempt"], smooth=sm)
    variants[sm] = a.group_by("nfl_id", "drill_name").agg(pl.col("ac_max", "ac_p90", "smax").max())
att = variants[3]


def index(att, drills, col="ac_max"):
    w = att.filter(pl.col("drill_name").is_in(drills)).pivot(on="drill_name", index="nfl_id", values=col)
    w = zcols(w, [c for c in w.columns if c != "nfl_id"])
    return w.select("nfl_id", pl.mean_horizontal([c for c in w.columns if c != "nfl_id"]).alias("idx"))


speed = index(att, CORE, "smax").rename({"idx": "speed_idx"})

# ---------- games: pressure (same outcome as composite_test) ----------
reg = data.reg_season_plays().join(dl_ids, on="nfl_id")
rush = reg.filter(pl.col("player_get_off").is_not_null()).with_columns(
    edge=pl.col("lined_up_position").is_in(["EDGE", "OLB"]).cast(pl.Int8),
    pressed=pl.col("time_to_pressure").is_not_null().cast(pl.Int8))
rush = rush.with_columns(press_adj=pl.col("pressed") - pl.col("pressed").mean().over("lined_up_position"))
g = (rush.group_by("nfl_id").agg(snaps=pl.len(), edge_share=pl.col("edge").mean(), s=pl.col("press_adj").sum(),
                                 pressures=pl.col("pressed").sum())
     .with_columns(press_adj=pl.col("s") / (pl.col("snaps") + 100)).filter(pl.col("snaps") >= 50))
base_df = (g.join(cr.select("nfl_id", "combine_weight", "ten_yd_split"), on="nfl_id").join(plr, on="nfl_id")
           .with_columns(log_pick=pl.col("draft_overall_pick").fill_null(260).log()))
BASE = ["log_pick", "combine_weight", "edge_share"]

print("\n[1] Robustness to drill choice")
for lab, drills in [("all 6 drills", CORE), ("only the 3 top-of-screen drills", SCREEN_TOP),
                    ("only the 3 drills that did NOT stand out (blind)", BLIND)]:
    fit(base_df.join(index(att, drills), on="nfl_id", how="left"), "press_adj", BASE, "idx", lab)
for d in CORE:
    fit(base_df.join(index(att, [d]), on="nfl_id", how="left"), "press_adj", BASE, "idx", f"  single drill: {d[:40]}")

print("\n[2] Not just speed")
full = base_df.join(index(att, CORE), on="nfl_id", how="left").join(speed, on="nfl_id", how="left")
fit(full, "press_adj", BASE, "speed_idx", "max speed alone")
fit(full, "press_adj", BASE + ["speed_idx"], "idx", "turn force | controlling for max speed")
for sm in (1, 5):
    fit(base_df.join(index(variants[sm], CORE), on="nfl_id", how="left"), "press_adj", BASE, "idx",
        f"smoothing {sm} frames")
fit(base_df.join(index(att, CORE, "ac_p90"), on="nfl_id", how="left"), "press_adj", BASE, "idx", "90th percentile instead of max")
fit(full, "press_adj", BASE + ["ten_yd_split"], "idx", "turn force | + 10-yd split")

# ---------- [2b] the same checks for the final CBS (3 arc drills) ----------
ARC = SCREEN_TOP
print("\n[2b] Final Combine Bend Score (3 arc drills)")
speed_arc = index(att, ARC, "smax").rename({"idx": "speed_idx"})
cbs = base_df.join(index(att, ARC), on="nfl_id", how="left").join(speed_arc, on="nfl_id", how="left")
fit(cbs, "press_adj", BASE, "idx", "CBS")
fit(cbs, "press_adj", BASE + ["speed_idx"], "idx", "CBS | controlling for max speed in the same drills")
for sm in (1, 5):
    fit(base_df.join(index(variants[sm], ARC), on="nfl_id", how="left"), "press_adj", BASE, "idx",
        f"CBS with smoothing {sm} frames")
fit(base_df.join(index(att, ARC, "ac_p90"), on="nfl_id", how="left"), "press_adj", BASE, "idx",
    "CBS from 90th percentile instead of peak")
pdf = cbs.select("press_adj", "idx", *BASE, "draft_year").drop_nulls().to_pandas()
for yr in sorted(pdf["draft_year"].unique()):
    tr, te = pdf[pdf["draft_year"] != yr], pdf[pdf["draft_year"] == yr]
    m0 = smf.ols(f"press_adj ~ {' + '.join(BASE)}", tr).fit()
    m1 = smf.ols(f"press_adj ~ {' + '.join(BASE)} + idx", tr).fit()
    e0, e1 = ((te.press_adj - m0.predict(te)) ** 2).mean(), ((te.press_adj - m1.predict(te)) ** 2).mean()
    print(f"  CBS holdout {yr}: n={len(te):2d}  beta_train={m1.params['idx']:+.4f}  MSE reduction={1 - e1 / e0:+.1%}")

# ---------- [3] turn force INSIDE games ----------
print("\n[3] In-game turn force (0-3 s after the snap, pass-rush snaps)")
keys = rush.select("game_id", "play_id", "nfl_id").unique()
frames = []
for yr in (2023, 2024, 2025):
    lf = (pl.scan_csv(data.RAW / f"game_tracking_{yr}.csv", null_values=data.NA, infer_schema_length=10000)
          .select("game_id", "play_id", "nfl_id", "time", "x", "y", "s", "event")
          .join(keys.lazy(), on=["game_id", "play_id", "nfl_id"]))
    frames.append(lf.collect())
gt = pl.concat(frames).with_columns(pl.col("time").str.to_datetime()).sort("game_id", "play_id", "nfl_id", "time")
snap = gt.filter(pl.col("event") == "ball_snap").group_by("game_id", "play_id", "nfl_id").agg(t0=pl.col("time").min())
gt = gt.join(snap, on=["game_id", "play_id", "nfl_id"]).with_columns(
    dt=(pl.col("time") - pl.col("t0")).dt.total_milliseconds() / 1000).filter(pl.col("dt").is_between(0, 3.0))
print(f"  rush frames: {gt.height:,}  rushes: {gt.select('game_id', 'play_id', 'nfl_id').n_unique():,}")
play_ac = centripetal(gt, ["game_id", "play_id", "nfl_id"], smooth=3)
play_ac.write_parquet(PROC / "dl_rush_play_turnforce.parquet")
game_tf = play_ac.group_by("nfl_id").agg(game_turn=pl.col("ac_max").mean(), game_turn_n=pl.len())
# play level: turn force on a rush vs pressure on that rush
pj = play_ac.join(rush.select("game_id", "play_id", "nfl_id", "pressed", "lined_up_position"),
                  on=["game_id", "play_id", "nfl_id"])
pdf = pj.to_pandas()
m = smf.logit("pressed ~ ac_max + C(lined_up_position)", pdf).fit(disp=0)
print(f"  play level: logit pressure ~ ac_max  beta={m.params['ac_max']:+.3f} p={m.pvalues['ac_max']:.2e} (n={len(pdf):,})")
tr = base_df.join(index(att, CORE), on="nfl_id", how="left").join(game_tf, on="nfl_id", how="left")
s = tr.select("idx", "game_turn").drop_nulls()
print(f"  Combine -> game transfer: r={np.corrcoef(s['idx'], s['game_turn'])[0, 1]:+.2f} (n={s.height})")
fit(tr, "game_turn", BASE, "idx", "in-game turn force ~ Combine index (with controls)")
fit(tr, "press_adj", BASE, "game_turn", "pressure ~ in-game turn force (with controls)")

tr.join(index(att, CORE).rename({"idx": "turn_force"}), on="nfl_id", how="left") \
  .write_parquet(PROC / "dl_player_table.parquet")
print("\nTop 12 by Combine turn force (with pick and pressure):")
with pl.Config(tbl_rows=12, tbl_width_chars=200):
    print(tr.sort("idx", descending=True).select("display_name", "nfl_position", "draft_overall_pick",
                                                 "idx", "pressures", "snaps", "press_adj").head(12))

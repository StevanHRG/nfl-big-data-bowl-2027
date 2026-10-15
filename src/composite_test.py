"""Composite indices (DL: turn force; WR: brake-and-burst), each the mean z-score across several drills.
Validation: OLS with controls, 5,000 permutations, and leave-one-draft-class-out prediction.
The WR idea is reported because it FAILED once its single strongest route was removed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
import data
from features import turn_features

rng = np.random.default_rng(27)
cr = data.combine_results()
pl_ = data.players().select("nfl_id", "draft_overall_pick", "draft_year")
reg = data.reg_season_plays()


def zmean(df: pl.DataFrame, cols: list[str], signs: dict[str, int]) -> pl.Series:
    z = [((pl.col(c) - pl.col(c).mean()) / pl.col(c).std() * signs.get(c, 1)) for c in cols]
    return df.select(pl.mean_horizontal(z)).to_series()


def per_drill(ct, ids, traits):
    f = turn_features(ct.join(ids, on="nfl_id"))
    return f.group_by("nfl_id", "drill_name").agg([pl.col(t).median() for t in traits])


def evaluate(df: pl.DataFrame, outcome: str, base: list[str], idx: str, label: str):
    d = df.select(outcome, idx, *base, "draft_year").drop_nulls().to_pandas()
    cols = [outcome, idx] + base
    z = d.copy(); z[cols] = (d[cols] - d[cols].mean()) / d[cols].std()
    f = f"{outcome} ~ {' + '.join(base)} + {idx}"
    m = smf.ols(f, z).fit()
    b_obs = m.params[idx]
    # permute the index: null distribution of the coefficient
    null = []
    for _ in range(5000):
        zz = z.copy(); zz[idx] = rng.permutation(zz[idx].values)
        null.append(smf.ols(f, zz).fit().params[idx])
    p_perm = (np.abs(null) >= abs(b_obs)).mean()
    # leave one draft class out
    gains = []
    for yr in sorted(d["draft_year"].unique()):
        tr, te = z[d["draft_year"] != yr], z[d["draft_year"] == yr]
        m0 = smf.ols(f"{outcome} ~ {' + '.join(base)}", tr).fit()
        m1 = smf.ols(f, tr).fit()
        e0 = ((te[outcome] - m0.predict(te)) ** 2).mean()
        e1 = ((te[outcome] - m1.predict(te)) ** 2).mean()
        gains.append((yr, len(te), 1 - e1 / e0, m1.params[idx]))
    print(f"\n=== {label} | n={len(d)} | {f}")
    print(f"beta={b_obs:+.3f} (z)  p_ols={m.pvalues[idx]:.4f}  p_perm={p_perm:.4f}  "
          f"CI95=[{m.conf_int().loc[idx, 0]:+.2f}, {m.conf_int().loc[idx, 1]:+.2f}]")
    for yr, n, g, b in gains:
        print(f"  holdout {yr}: n={n:2d}  MSE reduction={g:+.1%}  beta_train={b:+.2f}")


# ---------------- DL: turn force ----------------
dl_ids = cr.filter(pl.col("combine_position") == "DL").select("nfl_id")
DL_DRILLS = ["FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", "RUN_THE_HOOP_DRILL",
             "RUN_AND_CLUB_DRILL", "FOUR_BAG_AGILITY_DRILL", "PASS_RUSH_DRILL", "BODY_CONTROL_DRILL"]
f = per_drill(data.combine_tracking("SKILL_DRILLS_DL"), dl_ids, ["acel_c_max"]) \
    .filter(pl.col("drill_name").is_in(DL_DRILLS)).pivot(on="drill_name", index="nfl_id", values="acel_c_max")
f = f.with_columns(turn_force=zmean(f, DL_DRILLS, {}))
r = reg.join(dl_ids, on="nfl_id").filter(pl.col("player_get_off").is_not_null()).with_columns(
    edge=pl.col("lined_up_position").is_in(["EDGE", "OLB"]).cast(pl.Int8),
    pressed=pl.col("time_to_pressure").is_not_null().cast(pl.Int8))
r = r.with_columns(press_adj=pl.col("pressed") - pl.col("pressed").mean().over("lined_up_position"))
g = (r.group_by("nfl_id").agg(snaps=pl.len(), edge_share=pl.col("edge").mean(), s=pl.col("press_adj").sum())
     .with_columns(press_adj=pl.col("s") / (pl.col("snaps") + 100)).filter(pl.col("snaps") >= 50))
dl = (g.join(f.select("nfl_id", "turn_force"), on="nfl_id", how="left")
      .join(cr.select("nfl_id", "combine_weight", "ten_yd_split"), on="nfl_id").join(pl_, on="nfl_id")
      .with_columns(log_pick=pl.col("draft_overall_pick").fill_null(260).log()))
evaluate(dl, "press_adj", ["log_pick", "combine_weight", "edge_share"], "turn_force", "DL turn force")
evaluate(dl, "press_adj", ["log_pick", "combine_weight", "edge_share", "ten_yd_split"], "turn_force",
         "DL turn force (+10-yd split)")

# ---------------- WR: brake-and-burst ----------------
wr_ids = cr.filter(pl.col("combine_position") == "WR").select("nfl_id")
WR_BREAKS = ["DAGGER_ROUTE_LEFT", "CURL_ROUTE_RIGHT", "COMEBACK_ROUTE_RIGHT", "SLANT_ROUTE_LEFT",
             "SLOT_STRIKE_ROUTE_LEFT", "POST_CORNER_ROUTE_RIGHT", "SPEED_OUT_ROUTE_LEFT"]
fw = per_drill(data.combine_tracking("SKILL_DRILLS_WR"), wr_ids, ["decel", "reaccel", "min_s"]) \
    .filter(pl.col("drill_name").is_in(WR_BREAKS))
# per route: brake_burst = z(decel) + z(reaccel) - z(min_s), standardized within route
fw = fw.with_columns([((pl.col(c) - pl.col(c).mean().over("drill_name")) / pl.col(c).std().over("drill_name")).alias(c)
                      for c in ["decel", "reaccel", "min_s"]]) \
    .with_columns(bb=(pl.col("decel") + pl.col("reaccel") - pl.col("min_s")) / 3)
per_route = fw.pivot(on="drill_name", index="nfl_id", values="bb")
fw_all = fw.group_by("nfl_id").agg(brake_burst=pl.col("bb").mean(), n_routes=pl.len())
fw_nodag = fw.filter(pl.col("drill_name") != "DAGGER_ROUTE_LEFT").group_by("nfl_id").agg(brake_burst_nodagger=pl.col("bb").mean())
w = reg.join(wr_ids, on="nfl_id").filter(pl.col("separation_at_pass_forward").is_not_null())
w = w.with_columns(sep_adj=pl.col("separation_at_pass_forward") - pl.col("separation_at_pass_forward").mean().over("route_ran"))
gw = (w.group_by("nfl_id").agg(routes=pl.len(), s=pl.col("sep_adj").sum(),
                               slot_share=(pl.col("lined_up_position") == "SLOT_WR").mean())
      .with_columns(sep_adj=pl.col("s") / (pl.col("routes") + 50)).filter(pl.col("routes") >= 50))
wr = (gw.join(fw_all, on="nfl_id", how="left").join(fw_nodag, on="nfl_id", how="left")
      .join(cr.select("nfl_id", "combine_weight", "forty"), on="nfl_id").join(pl_, on="nfl_id")
      .with_columns(log_pick=pl.col("draft_overall_pick").fill_null(260).log()))
evaluate(wr, "sep_adj", ["log_pick", "combine_weight", "slot_share"], "brake_burst", "WR brake-and-burst (7 routes)")
evaluate(wr, "sep_adj", ["log_pick", "combine_weight", "slot_share"], "brake_burst_nodagger",
         "WR brake-and-burst WITHOUT the dagger route (robustness)")
evaluate(wr, "sep_adj", ["log_pick", "combine_weight", "slot_share", "forty"], "brake_burst",
         "WR brake-and-burst (+40-yd time)")
print("\nWR per-route correlation of brake_burst with sep_adj:")
pr = per_route.join(gw.select("nfl_id", "sep_adj"), on="nfl_id")
for c in WR_BREAKS:
    if c in pr.columns:
        s = pr.select(c, "sep_adj").drop_nulls()
        print(f"  {c:28s} n={s.height:3d} r={np.corrcoef(s[c], s['sep_adj'])[0, 1]:+.2f}")

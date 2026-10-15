"""Triagem: traços de curva/corte nos drills (DL e WR) vs. resultados em jogo, com controles."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
import data
from features import turn_features

TRAITS = ["acel_c_max", "bend_speed", "min_s", "post_s", "decel", "reaccel", "retain"]


def loo_r2(X, y):
    pred = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def screen(df: pl.DataFrame, outcome: str, base: list[str], label: str):
    pdf = df.to_pandas()
    cands = [c for c in df.columns if "__" in c]
    print(f"\n=== {label}: {outcome} | base: {' + '.join(base)} ===")
    hits = []
    for c in cands:
        d = pdf[[outcome, c] + base].dropna()
        if len(d) < 40:
            continue
        z = (d - d.mean()) / d.std()
        m = smf.ols(f"{outcome} ~ {' + '.join(base)} + Q('{c}')", z).fit()
        r0, r1 = loo_r2(z[base].values, z[outcome].values), loo_r2(z[base + [c]].values, z[outcome].values)
        hits.append((m.pvalues.iloc[-1], c, len(d), m.params.iloc[-1], r0, r1))
    hits.sort()
    print(f"{len(hits)} testes; Bonferroni 0.05 -> p < {0.05 / max(1, len(hits)):.4f}")
    for p, c, n, b, r0, r1 in hits[:8]:
        print(f"  {c:65s} n={n:3d} beta={b:+.2f} p={p:.4f} LOO-R2 {r0:.3f}->{r1:.3f}")


def drill_wide(ct, ids):
    f = turn_features(ct.join(ids, on="nfl_id"))
    agg = f.group_by("nfl_id", "drill_name").agg([pl.col(t).median() for t in TRAITS])
    out = None
    for t in TRAITS:
        w = agg.pivot(on="drill_name", index="nfl_id", values=t)
        w = w.rename({c: f"{t}__{c}" for c in w.columns if c != "nfl_id"})
        out = w if out is None else out.join(w, on="nfl_id", how="full", coalesce=True)
    return out


cr = data.combine_results()
pl_ = data.players().select("nfl_id", "draft_overall_pick")
reg = data.reg_season_plays()

# ---------- DL ----------
dl_ids = cr.filter(pl.col("combine_position") == "DL").select("nfl_id")
r = reg.join(dl_ids, on="nfl_id").filter(pl.col("player_get_off").is_not_null()).with_columns(
    edge=pl.col("lined_up_position").is_in(["EDGE", "OLB"]).cast(pl.Int8),
    pressed=pl.col("time_to_pressure").is_not_null().cast(pl.Int8))
r = r.with_columns(press_adj=pl.col("pressed") - pl.col("pressed").mean().over("lined_up_position"))
g = (r.group_by("nfl_id").agg(snaps=pl.len(), edge_share=pl.col("edge").mean(), s=pl.col("press_adj").sum())
     .with_columns(press_adj=pl.col("s") / (pl.col("snaps") + 100)).filter(pl.col("snaps") >= 50))
dl = (g.join(drill_wide(data.combine_tracking("SKILL_DRILLS_DL"), dl_ids), on="nfl_id", how="left")
      .join(cr.select("nfl_id", "combine_weight"), on="nfl_id").join(pl_, on="nfl_id")
      .with_columns(log_pick=pl.col("draft_overall_pick").fill_null(260).log()))
screen(dl, "press_adj", ["log_pick", "combine_weight", "edge_share"], "DL")

# ---------- WR ----------
wr_ids = cr.filter(pl.col("combine_position") == "WR").select("nfl_id")
w = reg.join(wr_ids, on="nfl_id").filter(pl.col("separation_at_pass_forward").is_not_null())
w = w.with_columns(sep_adj=pl.col("separation_at_pass_forward")
                   - pl.col("separation_at_pass_forward").mean().over("route_ran"))
gw = (w.group_by("nfl_id").agg(routes=pl.len(), s=pl.col("sep_adj").sum(),
                               slot_share=(pl.col("lined_up_position") == "SLOT_WR").mean())
      .with_columns(sep_adj=pl.col("s") / (pl.col("routes") + 50)).filter(pl.col("routes") >= 50))
wr = (gw.join(drill_wide(data.combine_tracking("SKILL_DRILLS_WR"), wr_ids), on="nfl_id", how="left")
      .join(cr.select("nfl_id", "combine_weight", "forty"), on="nfl_id").join(pl_, on="nfl_id")
      .with_columns(log_pick=pl.col("draft_overall_pick").fill_null(260).log()))
print(f"\nWR players: {wr.height}; sep_adj sd={wr['sep_adj'].std():.3f}")
screen(wr, "sep_adj", ["log_pick", "combine_weight", "slot_share"], "WR")

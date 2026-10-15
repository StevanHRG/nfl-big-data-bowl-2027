"""Rodada 2: resultados ajustados pela função em campo e teste incremental de cada traço do Combine."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
import data
from dl_viability import burst_features, DRILLS

cr = data.combine_results()
dl_ids = cr.filter(pl.col("combine_position") == "DL").select("nfl_id")

pp = data.reg_season_plays().join(dl_ids, on="nfl_id").filter(pl.col("player_get_off").is_not_null())
print("alignment on rush snaps:", pp["lined_up_position"].value_counts().sort("count", descending=True).rows())
pp = pp.with_columns(edge=(pl.col("lined_up_position").is_in(["EDGE", "OLB"])).cast(pl.Int8),
                     pressed=pl.col("time_to_pressure").is_not_null().cast(pl.Int8))
# ajuste por jogada: get-off e pressão relativos à média da função naquela jogada
for col in ["player_get_off", "pressed"]:
    pp = pp.with_columns((pl.col(col) - pl.col(col).mean().over("lined_up_position")).alias(f"{col}_adj"))
K = 100
game = (pp.group_by("nfl_id").agg(snaps=pl.len(), edge_share=pl.col("edge").mean(),
                                  go_adj_sum=pl.col("player_get_off_adj").sum(),
                                  pr_adj_sum=pl.col("pressed_adj").sum())
        .with_columns(get_off_adj=pl.col("go_adj_sum") / (pl.col("snaps") + 20),   # shrink a 0
                      press_adj=pl.col("pr_adj_sum") / (pl.col("snaps") + K))
        .filter(pl.col("snaps") >= 50))

ct = pl.concat([data.combine_tracking("SKILL_DRILLS_DL"), data.combine_tracking("FORTY_YARD_DASH")]) \
    .join(dl_ids, on="nfl_id").filter(pl.col("drill_name").is_in(DRILLS))
feat = burst_features(ct).group_by("nfl_id", "drill_name").agg(rise=pl.col("rise").min(), peak_a=pl.col("peak_a").max())
wide = feat.pivot(on="drill_name", index="nfl_id", values=["rise", "peak_a"])
df = (game.join(wide, on="nfl_id", how="left")
      .join(cr.select("nfl_id", "ten_yd_split", "combine_weight"), on="nfl_id")
      .join(data.players().select("nfl_id", "draft_overall_pick"), on="nfl_id")
      .with_columns(log_pick=pl.col("draft_overall_pick").fill_null(260).log()))
pdf = df.to_pandas()
base = ["log_pick", "combine_weight", "edge_share"]
cands = [c for c in df.columns if c.startswith(("rise_", "peak_a_"))] + ["ten_yd_split"]


def loo_r2(X, y):
    pred = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


for o in ["get_off_adj", "press_adj"]:
    print(f"\n=== outcome {o}  (base: {' + '.join(base)}) ===")
    print(f"{'feature':60s}  n  beta(z)   p     LOO-R2 base -> +feat")
    for c in cands:
        d = pdf[[o, c] + base].dropna()
        z = (d - d.mean()) / d.std()
        m = smf.ols(f"{o} ~ {' + '.join(base)} + {c}", z).fit()
        r0 = loo_r2(z[base].values, z[o].values)
        r1 = loo_r2(z[base + [c]].values, z[o].values)
        flag = " <--" if m.pvalues[c] < 0.05 and r1 > r0 else ""
        print(f"{c:60s} {len(d):3d}  {m.params[c]:+.2f}  {m.pvalues[c]:.3f}   {r0:.3f} -> {r1:.3f}{flag}")

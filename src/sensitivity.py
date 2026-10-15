"""Sensitivity of the CBS effect to arbitrary choices: shrinkage K, minimum rushes and the pick imputed for undrafted players."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
import data

tab = pl.read_parquet(data.PROCESSED / "dl_cbs_table.parquet").filter(pl.col("cbs").is_not_null(), pl.col("snaps").is_not_null())
print(f"{'K':>4} {'min':>4} {'udfa':>5}  n   CBS +1SD (POE/100)  95% CI           p")
for K in (50, 100, 200):
    for mn in (30, 50, 100):
        for udfa in (224, 260, 300):
            d = (tab.filter(pl.col("snaps") >= mn)
                 .with_columns(poe=100 * (pl.col("pressures") - pl.col("exp_press")) / (pl.col("snaps") + K),
                               lp=pl.col("draft_overall_pick").fill_null(udfa).log())
                 .to_pandas())
            m = smf.wls("poe ~ cbs + lp + combine_weight + edge_share", d, weights=d["snaps"]).fit()
            lo, hi = m.conf_int().loc["cbs"]
            print(f"{K:>4} {mn:>4} {udfa:>5} {len(d):3d}   {10*m.params['cbs']:+.2f}              "
                  f"[{10*lo:+.2f}, {10*hi:+.2f}]   {m.pvalues['cbs']:.4f}")

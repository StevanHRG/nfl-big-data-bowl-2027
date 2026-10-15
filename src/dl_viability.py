"""Failed idea #1 (reported for transparency): does pure burst in Combine DL drills relate to NFL get-off and pressure?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import polars as pl
from scipy import stats
import statsmodels.formula.api as smf
import data

LO, HI = 1.0, 4.0  # speed window (yd/s) used to time the burst
DRILLS = ["FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", "PASS_RUSH_DRILL", "FOUR_BAG_AGILITY_DRILL",
          "RUN_AND_CLUB_DRILL", "BODY_CONTROL_DRILL", "RUN_THE_HOOP_DRILL", "FORTY_YARD_DASH"]


def burst_features(ct: pl.DataFrame) -> pl.DataFrame:
    """Per attempt: time from first crossing LO to HI yd/s, and peak acceleration in that window."""
    ct = ct.with_columns(t=(pl.col("time") - pl.col("time").min()).over("nfl_id", "drill_name", "attempt")
                         .dt.total_milliseconds() / 1000)
    rows = []
    for (pid, drill, att), g in ct.group_by("nfl_id", "drill_name", "attempt"):
        s, a, t = g["s"].to_numpy(), g["a"].to_numpy(), g["t"].to_numpy()
        i_lo = np.argmax(s >= LO) if (s >= LO).any() else None
        if i_lo is None:
            continue
        after = np.nonzero(s[i_lo:] >= HI)[0]
        if not len(after):
            continue
        i_hi = i_lo + after[0]
        # linear interpolation to remove 0.1 s quantization
        def cross(i, lvl):
            if i == 0 or s[i] == s[i - 1]:
                return t[i]
            return t[i - 1] + (lvl - s[i - 1]) / (s[i] - s[i - 1]) * (t[i] - t[i - 1])
        rows.append(dict(nfl_id=pid, drill_name=drill, attempt=att,
                         rise=cross(i_hi, HI) - cross(i_lo, LO) if i_lo > 0 else None,
                         peak_a=a[i_lo:i_hi + 1].max(), smax=s.max()))
    return pl.DataFrame(rows)


if __name__ == "__main__":
    cr = data.combine_results()
    dl_ids = cr.filter(pl.col("combine_position") == "DL").select("nfl_id")
    ct = pl.concat([data.combine_tracking("SKILL_DRILLS_DL"), data.combine_tracking("FORTY_YARD_DASH")]) \
        .join(dl_ids, on="nfl_id").filter(pl.col("drill_name").is_in(DRILLS))
    att = burst_features(ct)
    print("attempts with burst:", att.height, " rise null (clip started moving):", att["rise"].null_count())
    feat = (att.group_by("nfl_id", "drill_name")
            .agg(rise=pl.col("rise").min(), peak_a=pl.col("peak_a").max(), smax=pl.col("smax").max()))
    wide = feat.pivot(on="drill_name", index="nfl_id", values=["rise", "peak_a"])

    # In-game outcomes (regular season)
    pp = data.reg_season_plays().join(dl_ids, on="nfl_id")
    rush = pp.filter(pl.col("player_get_off").is_not_null())
    game = rush.group_by("nfl_id").agg(
        snaps=pl.len(), get_off=pl.col("player_get_off").mean(),
        pressures=pl.col("time_to_pressure").is_not_null().sum(),
        sacks=pl.col("sack").fill_null(0).sum())
    lg_rate = game["pressures"].sum() / game["snaps"].sum()
    K = 100  # pseudo-rushes for shrinkage (simple empirical Bayes)
    game = game.with_columns(press_rate=(pl.col("pressures") + K * lg_rate) / (pl.col("snaps") + K))
    print(f"league pressure rate per rush snap (DL rookies): {lg_rate:.3f}")

    pl_ = data.players().select("nfl_id", "draft_overall_pick", "undrafted", "nfl_position")
    df = (game.filter(pl.col("snaps") >= 50)
          .join(wide, on="nfl_id", how="left")
          .join(cr.select("nfl_id", "ten_yd_split", "forty", "combine_weight", "ngs_athleticism_score"), on="nfl_id")
          .join(pl_, on="nfl_id")
          .with_columns(pick=pl.col("draft_overall_pick").fill_null(260)))
    print("players in analysis:", df.height, " undrafted:", df["undrafted"].sum())

    outcomes = ["get_off", "press_rate"]
    cands = [c for c in df.columns if c.startswith(("rise_", "peak_a_"))] + ["ten_yd_split", "forty", "ngs_athleticism_score", "pick"]
    print(f"\n{'feature':70s} n   rho(get_off)  rho(press_rate)")
    for c in cands:
        sub = df.select(c, *outcomes).drop_nulls()
        if sub.height < 30:
            continue
        r = [stats.spearmanr(sub[c], sub[o])[0] for o in outcomes]
        print(f"{c:70s} {sub.height:3d}  {r[0]:+.2f}         {r[1]:+.2f}")

    # Incremental test: does the burst trait add anything beyond pick, weight and the 10-yd split?
    pdf = df.to_pandas()
    pdf["log_pick"] = np.log(pdf["pick"])
    best = "rise_FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION"
    for o in outcomes:
        for base in ["log_pick + combine_weight", "log_pick + combine_weight + ten_yd_split"]:
            d = pdf[[o, best, "log_pick", "combine_weight", "ten_yd_split"]].dropna()
            z = (d - d.mean()) / d.std()
            m0 = smf.ols(f"{o} ~ {base}", z).fit()
            m1 = smf.ols(f"{o} ~ {base} + {best}", z).fit()
            print(f"\n[{o}] base: {base}  n={len(d)}  R2 {m0.rsquared:.3f} -> {m1.rsquared:.3f}  "
                  f"beta={m1.params[best]:+.2f} p={m1.pvalues[best]:.3f}")

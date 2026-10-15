"""Nested test: drills are selected using ONLY the training draft classes; the test class is never seen.
The same protocol is also run for offensive linemen (OL) as a comparison."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
import data
from features import centripetal

rng = np.random.default_rng(7)


def drill_table(drill_type, ids, min_players=60):
    ct = data.combine_tracking(drill_type).join(ids, on="nfl_id")
    keep = ct.group_by("drill_name").agg(pl.col("nfl_id").n_unique().alias("n")).filter(pl.col("n") >= min_players)
    ct = ct.join(keep.select("drill_name"), on="drill_name")
    a = centripetal(ct, ["nfl_id", "drill_name", "attempt"], smooth=3)
    return a.group_by("nfl_id", "drill_name").agg(pl.col("ac_max").max()) \
            .pivot(on="drill_name", index="nfl_id", values="ac_max")


def nested(df, drills, outcome, base, label, n_perm=300):
    pdf = df.to_pandas()
    years = sorted(pdf["draft_year"].unique())

    def run(p):
        preds, picks = [], []
        for yr in years:
            tr, te = p[p["draft_year"] != yr].copy(), p[p["draft_year"] == yr].copy()
            mu, sd = tr[drills].mean(), tr[drills].std()
            trz, tez = (tr[drills] - mu) / sd, (te[drills] - mu) / sd
            chosen = []
            for d in drills:
                dd = tr[[outcome] + base].assign(x=trz[d]).dropna()
                m = smf.ols(f"{outcome} ~ {' + '.join(base)} + x", dd).fit()
                if m.params["x"] > 0 and m.pvalues["x"] < 0.10:
                    chosen.append(d)
            if not chosen:  # blind fallback rule: use every drill
                chosen = drills
            picks.append((yr, len(chosen)))
            tr["idx"], te["idx"] = trz[chosen].mean(axis=1), tez[chosen].mean(axis=1)
            tr, te = tr.dropna(subset=[outcome, "idx"] + base), te.dropna(subset=[outcome, "idx"] + base)
            m0 = smf.ols(f"{outcome} ~ {' + '.join(base)}", tr).fit()
            m1 = smf.ols(f"{outcome} ~ {' + '.join(base)} + idx", tr).fit()
            preds.append(np.c_[te[outcome], m0.predict(te), m1.predict(te)])
        P = np.vstack(preds)
        return 1 - ((P[:, 0] - P[:, 2]) ** 2).sum() / ((P[:, 0] - P[:, 1]) ** 2).sum(), picks

    gain, picks = run(pdf)
    null = []
    for _ in range(n_perm):
        q = pdf.copy()
        q[drills] = q[drills].values[rng.permutation(len(q))]
        null.append(run(q)[0])
    p = (np.array(null) >= gain).mean()
    print(f"{label:34s} n={len(pdf):3d} drills={len(drills):2d}  out-of-sample MSE reduction={gain:+.1%}  "
          f"p_perm={p:.3f}  chosen per fold={picks}")
    return dict(gain=float(gain), p_perm=float(p), null=[float(v) for v in null])


if __name__ == "__main__":
    cr = data.combine_results()
    plr = data.players().select("nfl_id", "draft_overall_pick", "draft_year")
    reg = data.reg_season_plays()
    wt = cr.select("nfl_id", "combine_weight")

    def finish(g, ids, drill_type):
        tab = drill_table(drill_type, ids)
        drills = [c for c in tab.columns if c != "nfl_id"]
        df = (g.join(tab, on="nfl_id", how="left").join(wt, on="nfl_id").join(plr, on="nfl_id")
              .with_columns(log_pick=pl.col("draft_overall_pick").fill_null(260).log()))
        return df, drills

    # DL: pressure
    ids = cr.filter(pl.col("combine_position") == "DL").select("nfl_id")
    r = reg.join(ids, on="nfl_id").filter(pl.col("player_get_off").is_not_null()).with_columns(
        edge=pl.col("lined_up_position").is_in(["EDGE", "OLB"]).cast(pl.Int8),
        pressed=pl.col("time_to_pressure").is_not_null().cast(pl.Int8))
    r = r.with_columns(y=pl.col("pressed") - pl.col("pressed").mean().over("lined_up_position"))
    g = (r.group_by("nfl_id").agg(n=pl.len(), edge_share=pl.col("edge").mean(), s=pl.col("y").sum())
         .with_columns(y=pl.col("s") / (pl.col("n") + 100)).filter(pl.col("n") >= 50))
    df, drills = finish(g, ids, "SKILL_DRILLS_DL")
    res = nested(df, drills, "y", ["log_pick", "combine_weight", "edge_share"], "DL turn force -> pressure", n_perm=1000)
    import json
    (data.PROCESSED / "nested_dl.json").write_text(json.dumps(res))

    # OL: pressure allowed (lower is better -> use the negative)
    ids = cr.filter(pl.col("combine_position") == "OL").select("nfl_id")
    o = reg.join(ids, on="nfl_id").filter(pl.col("pressure_allowed").is_not_null()).with_columns(
        pa=pl.col("pressure_allowed").cast(pl.Int8))
    o = o.with_columns(y=-(pl.col("pa") - pl.col("pa").mean().over("lined_up_position")))
    g = (o.group_by("nfl_id").agg(n=pl.len(), tackle_share=(pl.col("lined_up_position") == "T").mean(),
                                  s=pl.col("y").sum())
         .with_columns(y=pl.col("s") / (pl.col("n") + 200)).filter(pl.col("n") >= 50))
    df, drills = finish(g, ids, "SKILL_DRILLS_OL")
    nested(df, drills, "y", ["log_pick", "combine_weight", "tackle_share"], "OL turn force -> pressure avoided")


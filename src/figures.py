"""Writeup figures (PNG, light theme). Colorblind-safe palette (first three categorical slots, validated all-pairs)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import data

FIG = data.ROOT / "figures"
FIG.mkdir(exist_ok=True)
PROC = data.PROCESSED

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#c43c3c", "#e88d8d", "#f0efec", "#86b6ef", "#1c5cab"])
ARC = ["FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", "RUN_THE_HOOP_DRILL", "RUN_AND_CLUB_DRILL"]

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK,
    "lines.linewidth": 2, "axes.titlepad": 28, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def subtitle(ax, text):
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(0, 7), textcoords="offset points",
                color=INK2, fontsize=10, va="bottom")


tab = pl.read_parquet(PROC / "dl_cbs_table.parquet")
an = tab.filter(pl.col("snaps") >= 50, pl.col("cbs").is_not_null()).to_pandas()


# ---------- Fig 1: what bend looks like (Run the Hoop) ----------
def fig1():
    ct = data.combine_tracking("SKILL_DRILLS_DL").filter(pl.col("drill_name") == "RUN_THE_HOOP_DRILL")
    has = tab.filter(pl.col("RUN_THE_HOOP_DRILL").is_not_null(), pl.col("snaps") >= 200)
    hi = has.sort("RUN_THE_HOOP_DRILL", descending=True).row(0, named=True)
    lo = has.sort("RUN_THE_HOOP_DRILL").row(0, named=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharex=True, sharey=True, layout="constrained")
    vmax = 9
    for ax, p, lab in [(axes[0], hi, "More turn force"), (axes[1], lo, "Less turn force")]:
        g = ct.filter(pl.col("nfl_id") == p["nfl_id"])
        att = g.group_by("attempt").agg(pl.len()).sort("len", descending=True)["attempt"][0]
        g = g.filter(pl.col("attempt") == att)
        x, y, s = g["x"].to_numpy(), g["y"].to_numpy(), g["s"].to_numpy()
        x, y = x - x[0], y - y[0]
        h = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
        ac = np.convolve(s * np.abs(np.gradient(h)) / 0.1, np.ones(3) / 3, mode="same")
        ax.plot(x, y, color=GRID, lw=2, zorder=1)
        sc = ax.scatter(x, y, c=ac, cmap=SEQ, vmin=0, vmax=vmax, s=46, edgecolor=SURFACE, linewidth=1.2, zorder=2)
        i = int(np.argmax(ac))
        ax.annotate(f"peak {ac[i]:.1f} yd/s²", (x[i], y[i]), xytext=(12, -16), textcoords="offset points",
                    fontsize=10, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        ax.set_aspect("equal")
        ax.set_title(f"{lab}: {p['display_name']}", fontsize=12)
        subtitle(ax, f"CBS {p['cbs']:.0f} · {p['pressures']} pressures on {p['snaps']} NFL pass rushes")
        ax.set_xlabel("yards")
    axes[0].set_ylabel("yards")
    cb = fig.colorbar(sc, ax=axes, shrink=0.75, pad=0.02)
    cb.set_label("centripetal acceleration (yd/s²)", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("Run the Hoop: the same arc to the eye, a different force to the sensor", x=0.02, ha="left", fontsize=15, fontweight="bold")
    fig.savefig(FIG / "fig1_hoop_paths.png")
    plt.close(fig)


# ---------- Fig 2: in-game mechanism ----------
def fig2():
    play = pl.read_parquet(PROC / "dl_rush_play_turnforce.parquet")
    reg = data.reg_season_plays().filter(pl.col("player_get_off").is_not_null()) \
        .select("game_id", "play_id", "nfl_id", "lined_up_position",
                pressed=pl.col("time_to_pressure").is_not_null().cast(pl.Int8))
    d = play.join(reg, on=["game_id", "play_id", "nfl_id"]).filter(
        pl.col("lined_up_position").is_in(["EDGE", "INTERIOR_LINE"]))
    edges = np.arange(0, 11, 1.0)
    fig, ax = plt.subplots(figsize=(9, 5))
    for pos, col, lab in [("EDGE", BLUE, "Edge"), ("INTERIOR_LINE", ORANGE, "Interior")]:
        g = d.filter(pl.col("lined_up_position") == pos).with_columns(
            b=pl.col("ac_max").clip(0, 10.99).floor()).group_by("b").agg(
            n=pl.len(), p=pl.col("pressed").mean()).filter(pl.col("n") >= 150).sort("b")
        xb, p, n = g["b"].to_numpy() + 0.5, g["p"].to_numpy(), g["n"].to_numpy()
        se = np.sqrt(p * (1 - p) / n)
        ax.fill_between(xb, 100 * (p - 1.96 * se), 100 * (p + 1.96 * se), color=col, alpha=0.15, lw=0)
        ax.plot(xb, 100 * p, color=col, marker="o", ms=8, mec=SURFACE, mew=2)
        ax.text(xb[-1] + 0.15, 100 * p[-1], lab, color=INK, va="center", fontsize=10)
    ax.set_xlim(0, 9.2)
    ax.set_xlabel("peak centripetal acceleration in the first 3 s after the snap (yd/s²)")
    ax.set_ylabel("pressures per 100 pass rushes")
    ax.set_title("On Sundays, rushers who turn harder reach the quarterback more")
    subtitle(ax, f"{d.height:,} rookie-contract DL pass rushes, 2023–25 regular season · band = 95% CI")
    ax.legend(handles=[mpl.lines.Line2D([], [], color=BLUE, marker="o", label="Edge"),
                       mpl.lines.Line2D([], [], color=ORANGE, marker="o", label="Interior")],
              frameon=False, loc="upper left")
    fig.savefig(FIG / "fig2_ingame_mechanism.png")
    plt.close(fig)


# ---------- Fig 3: main result by quartile ----------
def fig3():
    rng = np.random.default_rng(1)
    a = an.copy()
    a["q"] = np.digitize(a["cbs"], np.percentile(a["cbs"], [25, 50, 75])) + 1
    f = lambda d: 100 * (d["pressures"].sum() - d["exp_press"].sum()) / d["snaps"].sum()
    est = [f(a[a.q == q]) for q in range(1, 5)]
    boots = np.array([[f(s[s.q == q]) for q in range(1, 5)]
                      for s in (a.sample(len(a), replace=True, random_state=int(rng.integers(1e9))) for _ in range(2000))])
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    xs = np.arange(4)
    cols = [ORANGE if e < 0 else BLUE for e in est]
    ax.bar(xs, est, width=0.62, color=cols, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(xs, est, yerr=[np.array(est) - lo, hi - np.array(est)], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=4)
    for x, e in zip(xs, est):
        ax.text(x + 0.36, e, f"{e:+.1f}", ha="left", va="center", fontsize=11, color=INK, fontweight="bold")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(xs, ["Q1\nlowest CBS", "Q2", "Q3", "Q4\nhighest CBS"])
    ax.set_ylabel("pressures over expected\nper 100 pass rushes")
    ax.grid(axis="x", visible=False)
    ax.set_title("Higher Combine Bend Score, more NFL pressure")
    subtitle(ax, f"{len(a)} DL with ≥50 pass rushes · expected = alignment average (edge/interior) · bars = 95% bootstrap CI")
    fig.savefig(FIG / "fig3_quartiles.png")
    plt.close(fig)


# ---------- Fig 4: what it is worth (comparable coefficients) ----------
def fig4():
    d = an.dropna(subset=["ten_yd_split"]).copy()
    for c in ["cbs", "ten_yd_split", "log_pick", "combine_weight"]:
        d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std()
    d["ten_fast_z"] = -d["ten_yd_split_z"]   # faster = positive
    d["pick_early_z"] = -d["log_pick_z"]     # earlier pick = positive
    m = smf.wls("poe100 ~ cbs_z + ten_fast_z + pick_early_z + combine_weight_z + edge_share", d, weights=d["snaps"]).fit()
    rows = [("Combine Bend Score", "cbs_z", BLUE), ("Draft slot (earlier)", "pick_early_z", MUTED),
            ("10-yd split (faster)", "ten_fast_z", MUTED), ("Weight", "combine_weight_z", MUTED)]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, (lab, k, col) in enumerate(rows[::-1]):
        b, (lo, hi) = m.params[k], m.conf_int().loc[k]
        ax.plot([lo, hi], [i, i], color=col, lw=2)
        ax.plot(b, i, "o", color=col, ms=10, mec=SURFACE, mew=2)
        ax.text(hi + 0.08, i, f"{b:+.2f}", va="center", fontsize=10, color=INK)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.tick_params(axis="y", labelcolor=INK2, labelsize=11)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("pressures over expected per 100 pass rushes, per +1 SD")
    ax.set_title("Bend Score adds what the draft board and the stopwatch miss")
    subtitle(ax, f"Rush-weighted regression, n={int(m.nobs)} DL with a timed 10-yd split · lines = 95% CI")
    fig.savefig(FIG / "fig4_coefficients.png")
    plt.close(fig)


# ---------- Fig 5: out-of-sample validation ----------
def fig5():
    r = json.loads((PROC / "nested_dl.json").read_text())
    null = 100 * np.array(r["null"])
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.hist(null, bins=40, color="#b7d3f6", edgecolor=SURFACE, linewidth=1)
    ax.axvline(100 * r["gain"], color=BLUE, lw=2.5)
    ax.text(100 * r["gain"], ax.get_ylim()[1] * 0.92, f"  Actual CBS: {100*r['gain']:+.1f}%\n  p = {r['p_perm']:.3f}",
            color=INK, va="top", fontsize=11, fontweight="bold")
    ax.set_xlabel("improvement predicting pressure for an unseen draft class (% error reduction)")
    ax.set_ylabel("permutations")
    ax.grid(axis="x", visible=False)
    ax.set_title("Honest validation: pick drills blind, test on a held-out draft class")
    subtitle(ax, f"Drill selection uses training classes only · light bars = {len(null)} random shuffles")
    fig.savefig(FIG / "fig5_validation.png")
    plt.close(fig)


# ---------- Fig 6: scouting board ----------
def fig6():
    d = tab.filter(pl.col("cbs").is_not_null()).to_pandas()
    d["x"] = d["draft_overall_pick"].fillna(290)
    played = d[d["snaps"].fillna(0) >= 50]
    rest = d[d["snaps"].fillna(0) < 50]
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.scatter(rest["x"], rest["cbs"], s=40, facecolor="none", edgecolor=AXIS, linewidth=1.2, label="<50 NFL pass rushes")
    lim = np.nanmax(np.abs(played["poe100"]))
    sc = ax.scatter(played["x"], played["cbs"], c=played["poe100"], cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim),
                    s=70, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.axhline(60, color=AXIS, lw=1)
    ax.axvline(100, color=AXIS, lw=1)
    ax.text(103, d["cbs"].max() + 1.2, "Hidden-talent zone: CBS ≥ 60 and pick > 100", color=INK2, fontsize=10)
    offsets = {"Cedric Johnson": (10, 18), "Javon Solomon": (-30, 26), "Adetomiwa Adebawore": (-14, -24),
               "Robert Beal": (25, -30), "Laiatu Latu": (14, 6), "Chop Robinson": (12, -16), "Tuli Tuipulotu": (14, 4)}
    for name, (dx, dy) in offsets.items():
        r = d[d["display_name"] == name]
        if len(r):
            r = r.iloc[0]
            label = "A. Adebawore" if name == "Adetomiwa Adebawore" else name
            ax.annotate(label, (r["x"], r["cbs"]), xytext=(dx, dy), textcoords="offset points", fontsize=9, color=INK,
                        ha="right" if dx < 0 else "left", va="center",
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=0, shrinkB=4))
    ax.set_xticks([1, 32, 64, 100, 150, 200, 250, 290], ["1", "32", "64", "100", "150", "200", "250", "UDFA"])
    ax.set_xlabel("overall draft pick")
    ax.set_ylabel("Combine Bend Score (50 = average)")
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("pressures over expected per 100 pass rushes", color=INK2)
    cb.outline.set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Scouting board: Bend Score vs. draft slot")
    subtitle(ax, "Each dot is a 2023–25 DL prospect · blue = pressured above expectation, red = below")
    fig.savefig(FIG / "fig6_scouting_board.png")
    plt.close(fig)


# ---------- Fig 1 (animated): the same two Run the Hoop attempts in real time ----------
def fig1gif():
    from matplotlib.animation import FuncAnimation, PillowWriter
    ct = data.combine_tracking("SKILL_DRILLS_DL").filter(pl.col("drill_name") == "RUN_THE_HOOP_DRILL")
    has = tab.filter(pl.col("RUN_THE_HOOP_DRILL").is_not_null(), pl.col("snaps") >= 200)
    players = [has.sort("RUN_THE_HOOP_DRILL", descending=True).row(0, named=True),
               has.sort("RUN_THE_HOOP_DRILL").row(0, named=True)]
    tracks = []
    for p in players:
        g = ct.filter(pl.col("nfl_id") == p["nfl_id"])
        att = g.group_by("attempt").agg(pl.len()).sort("len", descending=True)["attempt"][0]
        g = g.filter(pl.col("attempt") == att)
        x, y, s = g["x"].to_numpy(), g["y"].to_numpy(), g["s"].to_numpy()
        x, y = x - x[0], y - y[0]
        h = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
        ac = np.convolve(s * np.abs(np.gradient(h)) / 0.1, np.ones(3) / 3, mode="same")
        tracks.append((x, y, ac))
    n = max(len(t[0]) for t in tracks)
    hold = 15  # 1.5 s pause on the final frame
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True, layout="constrained")
    xs = np.concatenate([t[0] for t in tracks]); ys = np.concatenate([t[1] for t in tracks])
    arts = []
    for ax, p, (x, y, ac), lab in zip(axes, players, tracks, ["More turn force", "Less turn force"]):
        ax.set_xlim(xs.min() - 0.8, xs.max() + 0.8); ax.set_ylim(ys.min() - 0.8, ys.max() + 0.8)
        ax.set_aspect("equal")
        ax.plot(x, y, color=GRID, lw=2, zorder=1)
        ax.set_title(f"{lab}: {p['display_name']}", fontsize=12)
        subtitle(ax, f"CBS {p['cbs']:.0f} · {p['pressures']} pressures on {p['snaps']} NFL pass rushes")
        ax.set_xlabel("yards")
        sc = ax.scatter([], [], c=[], cmap=SEQ, vmin=0, vmax=9, s=46, edgecolor=SURFACE, linewidth=1.2, zorder=2)
        head, = ax.plot([], [], "o", ms=13, color=INK, mec=SURFACE, mew=2, zorder=3)
        txt = ax.text(0.97, 0.05, "", transform=ax.transAxes, fontsize=11, color=INK, fontweight="bold", ha="right")
        arts.append((sc, head, txt, x, y, ac))
    axes[0].set_ylabel("yards")
    cb = fig.colorbar(arts[0][0], ax=axes, shrink=0.75, pad=0.02)
    cb.set_label("centripetal acceleration (yd/s²)", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("Run the Hoop in real time: same arc, different force", x=0.02, ha="left", fontsize=14, fontweight="bold")

    def update(i):
        out = []
        for sc, head, txt, x, y, ac in arts:
            k = min(i, len(x) - 1) + 1
            sc.set_offsets(np.c_[x[:k], y[:k]]); sc.set_array(ac[:k])
            head.set_data([x[k - 1]], [y[k - 1]])
            txt.set_text(f"now {ac[k - 1]:.1f}  ·  peak {ac[:k].max():.1f} yd/s²")
            out += [sc, head, txt]
        return out

    anim = FuncAnimation(fig, update, frames=n + hold, blit=False)
    anim.save(FIG / "fig1_hoop_paths.gif", writer=PillowWriter(fps=10), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "1gif", "2", "3", "4", "5", "6"]
    for w in which:
        globals()[f"fig{w}"]()
        print("ok fig", w)

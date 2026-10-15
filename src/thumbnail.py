"""Card/thumbnail image for the Kaggle Writeup (560 x 280 px)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import data
from figures import FIG, SEQ, SURFACE, INK, INK2, GRID, tab

ct = data.combine_tracking("SKILL_DRILLS_DL").filter(pl.col("drill_name") == "RUN_THE_HOOP_DRILL")
p = tab.filter(pl.col("RUN_THE_HOOP_DRILL").is_not_null(), pl.col("snaps") >= 200) \
       .sort("RUN_THE_HOOP_DRILL", descending=True).row(0, named=True)
g = ct.filter(pl.col("nfl_id") == p["nfl_id"])
att = g.group_by("attempt").agg(pl.len()).sort("len", descending=True)["attempt"][0]
g = g.filter(pl.col("attempt") == att)
x, y, s = g["x"].to_numpy(), g["y"].to_numpy(), g["s"].to_numpy()
x, y = x - x[0], y - y[0]
h = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
ac = np.convolve(s * np.abs(np.gradient(h)) / 0.1, np.ones(3) / 3, mode="same")

plt.rcParams["savefig.bbox"] = "standard"  # exact pixel size, no tight crop
fig = plt.figure(figsize=(5.6, 2.8), dpi=100)
fig.patch.set_facecolor(SURFACE)
ax = fig.add_axes([0.50, 0.04, 0.48, 0.92])
ax.plot(x, y, color=GRID, lw=2, zorder=1)
ax.scatter(x, y, c=ac, cmap=SEQ, vmin=0, vmax=9, s=22, edgecolor=SURFACE, linewidth=0.8, zorder=2)
ax.set_aspect("equal"); ax.axis("off")
fig.text(0.05, 0.70, "Bend You\nCan Measure", fontsize=22, fontweight="bold", color=INK, va="center", linespacing=1.0)
fig.text(0.05, 0.38, "Combine turn force predicts\nNFL pass-rush pressure", fontsize=11, color=INK2, va="center")
fig.text(0.05, 0.14, "+1 SD → +1 pressure / 100 rushes", fontsize=10, color="#256abf", fontweight="bold", va="center")
fig.savefig(FIG / "thumbnail_560x280.png", dpi=100, facecolor=SURFACE)
print("ok")

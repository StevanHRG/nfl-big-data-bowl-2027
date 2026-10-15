"""Per-attempt movement traits from drill tracking: turning force (bend) and brake/re-acceleration at the cut."""
import numpy as np
import polars as pl


def _angdiff(d):
    """Frame-to-frame angle difference in degrees, wrapped to [-180, 180)."""
    return (np.diff(d) + 180) % 360 - 180


def turn_features(ct: pl.DataFrame) -> pl.DataFrame:
    """Per attempt (nfl_id, drill_name, attempt):
    - acel_c_max: peak sustained centripetal acceleration (s * omega, 3-frame moving average), yd/s²
    - bend_speed: mean speed on frames with a strong turn (omega > 60 deg/s and s > 2)
    - pre_s, min_s, post_s: speed before the main cut, at the cut and 1 s after
    - decel: speed lost per second into the cut; reaccel: speed gained per second after it
    - retain: post_s / pre_s
    """
    rows = []
    for (pid, drill, att), g in ct.group_by("nfl_id", "drill_name", "attempt"):
        s, d = g["s"].to_numpy(), g["dir"].to_numpy()
        if len(s) < 15:
            continue
        x, y = g["x"].to_numpy(), g["y"].to_numpy()
        # heading from the x/y path (more stable than 'dir' at low speed)
        heading = np.degrees(np.arctan2(np.gradient(y), np.gradient(x)))
        omega = np.abs(_angdiff(heading)) / 0.1  # graus/s
        omega = np.convolve(omega, np.ones(3) / 3, mode="same")
        sm = s[1:]
        ac = sm * np.radians(omega)
        curving = (omega > 60) & (sm > 2)
        # main cut: largest cumulative heading change within a 0.5 s window
        win = 5
        cum = np.array([np.abs(_angdiff(heading[i:i + win + 1])).sum() for i in range(len(heading) - win)])
        i_cut = int(np.argmax(cum)) + win // 2
        i_pre0, i_post1 = max(0, i_cut - 10), min(len(s) - 1, i_cut + 10)
        pre_s = s[i_pre0:i_cut + 1].max()
        i_min = i_pre0 + int(np.argmin(s[i_pre0:i_post1 + 1]))
        min_s = s[i_min]
        i_peak_pre = i_pre0 + int(np.argmax(s[i_pre0:i_min + 1]))
        post_s = s[min(len(s) - 1, i_min + 10)]
        rows.append(dict(
            nfl_id=pid, drill_name=drill, attempt=att,
            acel_c_max=float(np.max(np.convolve(ac, np.ones(3) / 3, mode="valid"))) if len(ac) >= 3 else None,
            bend_speed=float(sm[curving].mean()) if curving.sum() >= 3 else None,
            cut_angle=float(cum.max()),
            pre_s=float(pre_s), min_s=float(min_s), post_s=float(post_s),
            decel=float((pre_s - min_s) / max(0.1, (i_min - i_peak_pre) * 0.1)),
            reaccel=float((post_s - min_s) / 1.0),
            retain=float(post_s / pre_s) if pre_s > 0 else None,
        ))
    return pl.DataFrame(rows)


def centripetal(df: pl.DataFrame, keys: list[str], smooth: int = 3) -> pl.DataFrame:
    """Per-frame centripetal acceleration (yd/s²) from x, y; frames must be time-ordered within keys."""
    out = []
    for k, g in df.group_by(keys):
        x, y, s = g["x"].to_numpy(), g["y"].to_numpy(), g["s"].to_numpy()
        if len(x) < 6:
            continue
        h = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
        omega = np.abs(np.gradient(h)) / 0.1
        ac = np.convolve(s * omega, np.ones(smooth) / smooth, mode="same")
        out.append(dict(zip(keys, k)) | dict(ac_max=float(ac.max()), ac_p90=float(np.percentile(ac, 90)),
                                             smax=float(s.max())))
    return pl.DataFrame(out)


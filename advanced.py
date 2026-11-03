"""Step 2b: advanced metrics from tracking data, written to runs/<name>/advanced.json.

1. Pitch control per frame (Spearman's model, Shaw's simplification): each player keeps their
   velocity for a 0.7 s reaction time, then runs at 5 m/s; a cell goes to the team that gets
   there first, softened by a logistic with sigma 0.45 s. Only cells the camera sees count.
2. Pressure on the ball carrier: opponents within 5 m and pressing intensity, the expected
   number of opponents able to reach the carrier within 1.5 s (after Bekkers 2025).
3. Expected Threat (Karun Singh's open 12 x 8 grid): xT added by every pass.
4. Packing: opponents a completed pass takes out of the game.
5. Team shape per frame: width, depth, convex hull area and defensive line height.
6. Pitch to pixel homography per frame, so the page can draw all of this on the real grass.

    python advanced.py runs/match
"""
import argparse
import base64
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from pitch import LENGTH, WIDTH

REACTION, VMAX, SIGMA = 0.7, 5.0, 0.45
PRESS_WINDOW = 1.5
GX, GY = 42, 28                      # 2.5 m cells
K = math.pi / (math.sqrt(3) * SIGMA)

XT = np.array([  # Karun Singh, open xT 12 x 8 v1 (rows = width, cols = length, attacking right)
    [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267, 0.01248344, 0.01473596, 0.0174506, 0.02122129, 0.02756312, 0.03485072, 0.0379259],
    [0.00750072, 0.00878589, 0.00942382, 0.0105949, 0.01214719, 0.0138454, 0.01611813, 0.01870347, 0.02401521, 0.02953272, 0.04066992, 0.04647721],
    [0.0088799, 0.00977745, 0.01001304, 0.01110462, 0.01269174, 0.01429128, 0.01685596, 0.01935132, 0.0241224, 0.02855202, 0.05491138, 0.06442595],
    [0.00941056, 0.01082722, 0.01016549, 0.01132376, 0.01262646, 0.01484598, 0.01689528, 0.0199707, 0.02385149, 0.03511326, 0.10805102, 0.25745362],
    [0.00941056, 0.01082722, 0.01016549, 0.01132376, 0.01262646, 0.01484598, 0.01689528, 0.0199707, 0.02385149, 0.03511326, 0.10805102, 0.25745362],
    [0.0088799, 0.00977745, 0.01001304, 0.01110462, 0.01269174, 0.01429128, 0.01685596, 0.01935132, 0.0241224, 0.02855202, 0.05491138, 0.06442595],
    [0.00750072, 0.00878589, 0.00942382, 0.0105949, 0.01214719, 0.0138454, 0.01611813, 0.01870347, 0.02401521, 0.02953272, 0.04066992, 0.04647721],
    [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267, 0.01248344, 0.01473596, 0.0174506, 0.02122129, 0.02756312, 0.03485072, 0.0379259],
])


def xt(x, y):
    return float(XT[min(7, max(0, int(y / WIDTH * 8))), min(11, max(0, int(x / LENGTH * 12)))])


def hull_area(pts):
    """Monotone chain convex hull, then the shoelace formula."""
    pts = sorted(map(tuple, pts))
    if len(pts) < 3:
        return 0.0
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    h = np.array(lower[:-1] + upper[:-1])
    return float(0.5 * abs(np.dot(h[:, 0], np.roll(h[:, 1], 1)) - np.dot(h[:, 1], np.roll(h[:, 0], 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    args = ap.parse_args()
    run = Path(args.run)
    tr = json.loads((run / "tracks.json").read_text())
    st = json.loads((run / "stats.json").read_text())
    frames, tracks = tr["frames"], tr["tracks"]
    W, H = tr["width"], tr["height"]
    live = st["live"]
    t = np.array(live["t"])
    n = len(frames)
    right = live["attacks_right"]                      # does team k attack the right goal

    # ---- velocities per track (centred difference over about 0.6 s) ------------------
    series = defaultdict(list)
    for i, f in enumerate(frames):
        for p in f["people"]:
            if "xy" in p:
                series[p["id"]].append((i, *p["xy"]))
    vel = {}
    for tid, s in series.items():
        a = np.array(s)
        for j in range(len(a)):
            lo, hi = max(0, j - 2), min(len(a) - 1, j + 2)
            dt = t[int(a[hi, 0])] - t[int(a[lo, 0])]
            v = (a[hi, 1:] - a[lo, 1:]) / dt if dt > 0 and dt < 1.2 else np.zeros(2)
            sp = np.linalg.norm(v)
            if sp > 10:
                v = v / sp * 10
            vel[(tid, int(a[j, 0]))] = v

    cx = (np.arange(GX) + .5) * LENGTH / GX
    cy = (np.arange(GY) + .5) * WIDTH / GY
    CX, CY = np.meshgrid(cx, cy)                       # rows = y, cols = x
    cells = np.stack([CX.ravel(), CY.ravel()], 1)

    pc_bytes = bytearray()
    hinv, space, holder, press_n, press_i, press_d = [], [[], []], [], [], [], []
    shape = {k: [[], []] for k in ("width", "depth", "area", "line")}
    pc_sum = [np.zeros(GX * GY), np.zeros(GX * GY)]
    pc_cnt = np.zeros(GX * GY)

    for i, f in enumerate(frames):
        people = []
        src, dst = [], []
        for p in f["people"]:
            if "xy" not in p:
                continue
            x1, y1, x2, y2 = p["box"]
            src.append(p["xy"])
            dst.append([(x1 + x2) / 2, y2])
            info = tracks[str(p["id"])]
            if info["role"] != "referee" and info["team"] >= 0:
                people.append((p["id"], info["team"], info["role"], np.array(p["xy"]), vel.get((p["id"], i), np.zeros(2))))

        # pitch -> pixel homography, refit from the feet positions it produced
        Hp = None
        if len(src) >= 4:
            Hp, _ = cv2.findHomography(np.float32(src), np.float32(dst), 0)
        if Hp is not None and abs(Hp[2, 2]) > 1e-9:
            Hp = Hp / Hp[2, 2]
            hinv.append([round(float(v), 6) for v in Hp.ravel()])
            uvw = np.c_[cells, np.ones(len(cells))] @ Hp.T
            w = uvw[:, 2]
            u, v = uvw[:, 0] / np.where(w == 0, 1, w), uvw[:, 1] / np.where(w == 0, 1, w)
            visible = (w > 0) & (u >= 0) & (u <= W) & (v >= 0) & (v <= H)
        else:
            hinv.append(None)
            visible = np.zeros(len(cells), bool)

        # ---- pitch control -----------------------------------------------------------
        T = []
        for k in (0, 1):
            pts = [(xy + v * REACTION) for (_, team, _, xy, v) in people if team == k]
            if not pts:
                T.append(None)
                continue
            R = np.array(pts)
            d = np.sqrt(((cells[:, None, :] - R[None, :, :]) ** 2).sum(-1))
            T.append((REACTION + d / VMAX).min(1))
        if T[0] is not None and T[1] is not None and visible.any():
            P0 = 1 / (1 + np.exp(-K * (T[1] - T[0])))
            enc = np.where(visible, 1 + np.round(P0 * 254), 0).astype(np.uint8)
            space[0].append(round(float(P0[visible].mean() * 100), 1))
            space[1].append(round(float((1 - P0[visible]).mean() * 100), 1))
            pc_sum[0] += np.where(visible, P0, 0)
            pc_sum[1] += np.where(visible, 1 - P0, 0)
            pc_cnt += visible
        else:
            enc = np.zeros(len(cells), np.uint8)
            space[0].append(None)
            space[1].append(None)
        pc_bytes += enc.tobytes()

        # ---- ball carrier and pressure ------------------------------------------------
        k = live["poss"][i]
        b = live["ball"][i]
        hold, best = None, 3.0
        if k is not None and b is not None:
            for (tid, team, _, xy, _) in people:
                d = float(np.linalg.norm(xy - b))
                if team == k and d < best:
                    hold, best = (tid, xy), d
        holder.append(hold[0] if hold else None)
        if hold:
            opp = [(xy, v) for (_, team, role, xy, v) in people if team != k]
            dists = [float(np.linalg.norm(xy - hold[1])) for xy, _ in opp]
            tti = [REACTION + np.linalg.norm(hold[1] - (xy + v * REACTION)) / VMAX for xy, v in opp]
            press_n.append(sum(d < 5 for d in dists))
            press_d.append(round(min(dists), 1) if dists else None)
            press_i.append(round(float(sum(1 / (1 + math.exp(-K * (PRESS_WINDOW - x))) for x in tti)), 2))
        else:
            press_n.append(None)
            press_d.append(None)
            press_i.append(None)

        # ---- team shape (outfield players on screen) -------------------------------------
        for kk in (0, 1):
            pts = np.array([xy for (_, team, role, xy, _) in people if team == kk and role == "player"])
            if len(pts) >= 4:
                shape["width"][kk].append(round(float(pts[:, 1].max() - pts[:, 1].min()), 1))
                shape["depth"][kk].append(round(float(pts[:, 0].max() - pts[:, 0].min()), 1))
                shape["area"][kk].append(round(hull_area(pts)))
                own = pts[:, 0].min() if right[kk] else LENGTH - pts[:, 0].max()
                shape["line"][kk].append(round(float(own), 1))
            else:
                for key in shape:
                    shape[key][kk].append(None)

    # ---- passes: xT, packing, pressure on the passer, space at the target --------------
    def frame_at(sec):
        return int(np.argmin(np.abs(t - sec)))

    def opp_ahead(i, team, x_norm):
        """Opponents (incl. keeper) closer to the target goal than x_norm, in team's attacking frame."""
        cnt = 0
        for p in frames[i]["people"]:
            info = tracks[str(p["id"])]
            if "xy" in p and info["team"] not in (-1, team) and info["role"] != "referee":
                x = p["xy"][0] if right[team] else LENGTH - p["xy"][0]
                cnt += x > x_norm
        return cnt

    pc_all = np.frombuffer(bytes(pc_bytes), np.uint8).reshape(n, GY, GX)
    extras = []
    for p in st["passes"]:
        k = p["team"]
        i0, i1 = frame_at(p["t"]), frame_at(p["t2"])
        value = (xt(p["x2"], p["y2"]) - xt(p["x1"], p["y1"])) if p["ok"] else -xt(p["x1"], p["y1"])
        packing = max(0, opp_ahead(i0, k, p["x1"]) - opp_ahead(i1, k, p["x2"])) if p["ok"] and p["len"] >= 5 else 0   # shorter "passes" are tracking noise
        passer = np.array(p["raw"][:2])
        near = [np.linalg.norm(np.array(q["xy"]) - passer) for q in frames[i0]["people"]
                if "xy" in q and tracks[str(q["id"])]["team"] not in (-1, k) and tracks[str(q["id"])]["role"] != "referee"]
        cell = pc_all[i1, min(GY - 1, int(p["raw"][3] / WIDTH * GY)), min(GX - 1, int(p["raw"][2] / LENGTH * GX))]
        ctrl = None if cell == 0 else round(((cell - 1) / 254 if k == 0 else 1 - (cell - 1) / 254) * 100)
        extras.append({"xt": round(value, 4), "packing": int(packing),
                       "pressured": bool(near and min(near) < 3), "space_at_target": ctrl})

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(float(np.mean(xs)), 1) if xs else None

    teams = []
    for k in (0, 1):
        idx = [j for j, p in enumerate(st["passes"]) if p["team"] == k]
        ok = [j for j in idx if st["passes"][j]["ok"]]
        hold_frames = [j for j in range(n) if live["poss"][j] == (1 - k) and press_i[j] is not None]
        teams.append({
            "xt": round(sum(extras[j]["xt"] for j in idx), 3),
            "xt_positive": round(sum(max(0, extras[j]["xt"]) for j in idx), 3),
            "packing": sum(extras[j]["packing"] for j in ok),
            "packing_per_pass": round(sum(extras[j]["packing"] for j in ok) / len(ok), 2) if ok else 0,
            "line_breaking": sum(extras[j]["packing"] >= 2 for j in ok),
            "under_pressure": round(sum(extras[j]["pressured"] for j in idx) / len(idx) * 100) if idx else 0,
            "space": mean(space[k]),
            "pressing": mean([press_i[j] for j in hold_frames]),
            "width": mean(shape["width"][k]), "depth": mean(shape["depth"][k]),
            "area": mean(shape["area"][k]), "line": mean(shape["line"][k]),
        })

    player_xt = defaultdict(float)
    for p, e in zip(st["passes"], extras):
        player_xt[p["from"]] += e["xt"]
    with np.errstate(invalid="ignore"):
        avg = [np.where(pc_cnt > 0, pc_sum[k] / np.maximum(pc_cnt, 1), -1).reshape(GY, GX).round(3).tolist() for k in (0, 1)]

    out = {
        "params": {"reaction_s": REACTION, "vmax_ms": VMAX, "sigma_s": SIGMA, "press_window_s": PRESS_WINDOW},
        "pc": {"grid": [GX, GY], "data": base64.b64encode(bytes(pc_bytes)).decode(), "avg": avg},
        "hinv": hinv,
        "live": {"space": space, "holder": holder, "press_n": press_n, "press_i": press_i, "press_d": press_d,
                 "shape": shape},
        "passes": extras, "teams": teams, "xt_grid": XT.round(4).tolist(),
        "player_xt": sorted([[pid, round(v, 3)] for pid, v in player_xt.items()], key=lambda r: -r[1]),
    }
    (run / "advanced.json").write_text(json.dumps(out, separators=(",", ":")))
    for k, tm in enumerate(teams):
        print(f"{st['teams'][k]['name']}: xT {tm['xt']}, packing {tm['packing']} ({tm['line_breaking']} line breaking), "
              f"space {tm['space']}%, width {tm['width']} m, depth {tm['depth']} m, line height {tm['line']} m, "
              f"pressing {tm['pressing']}")
    print(f"homography for {sum(h is not None for h in hinv)}/{n} frames, wrote {run / 'advanced.json'}")


if __name__ == "__main__":
    main()

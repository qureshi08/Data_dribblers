"""Step 2: tracking data in, match analytics out.

Possession, passes, shots with xG, distance and speed per player, heatmaps,
average positions and the pass network, all computed from runs/<name>/tracks.json.

    python stats.py runs/match
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pitch import GOAL_W, LENGTH, WIDTH

CONTROL_DIST = 2.5      # metres between ball and feet to count as on the ball
MAX_RUN_SPEED = 12.0    # m/s, anything faster is a tracking jump
SPRINT_SPEED = 7.0      # m/s (25 km/h)
SHOT_SPEED = 13.0       # m/s
SHOT_RANGE = 35.0       # metres from goal
GRID = (24, 16)


def interpolate_ball(frames, max_gap_s):
    t = np.array([f["t"] for f in frames])
    xy = np.full((len(frames), 2), np.nan)
    for i, f in enumerate(frames):
        if f["ball"] and "xy" in f["ball"]:
            xy[i] = f["ball"]["xy"]
    # drop single frame teleports
    ok = ~np.isnan(xy[:, 0])
    idx = np.where(ok)[0]
    for a, b, c in zip(idx, idx[1:], idx[2:]):
        if np.linalg.norm(xy[b] - xy[a]) / (t[b] - t[a]) > 40 and np.linalg.norm(xy[c] - xy[a]) / (t[c] - t[a]) < 40:
            xy[b] = np.nan
    ok = ~np.isnan(xy[:, 0])
    idx = np.where(ok)[0]
    observed = ok.copy()
    for a, b in zip(idx, idx[1:]):
        if 1 < b - a and t[b] - t[a] <= max_gap_s:
            for k in range(a + 1, b):
                w = (t[k] - t[a]) / (t[b] - t[a])
                xy[k] = xy[a] * (1 - w) + xy[b] * w
    return t, xy, observed


def xg(x_to_goal, y_off):
    """Simple distance and angle model, roughly 0.3 from the penalty spot and 0.04 from 25 m."""
    d = math.hypot(x_to_goal, y_off)
    angle = math.atan2(GOAL_W * x_to_goal, x_to_goal ** 2 + y_off ** 2 - (GOAL_W / 2) ** 2)
    if angle < 0:
        angle += math.pi
    return 1 / (1 + math.exp(-(-0.85 + 2.0 * angle - 0.1164 * d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--halftime", type=float, default=0, help="second the second half starts, if the video has both")
    ap.add_argument("--names", default="Team A,Team B")
    args = ap.parse_args()
    run = Path(args.run)
    data = json.loads((run / "tracks.json").read_text())
    frames, fps, stride = data["frames"], data["fps"], data["stride"]
    # Video space mode: a camera the pitch model cannot map (an operator panning over a futsal court) still gives
    # possession, passes, touches and ball wins. Positions come from the frame, scaled so "on the ball" means the
    # ball is within about half a player's height of their feet. Shots, forward passes and running stats need real
    # pitch geometry, so they are switched off instead of guessed.
    IMAGE = sum(f["h"] for f in frames) < .3 * max(1, len(frames))
    if IMAGE:
        hs = [p["box"][3] - p["box"][1] for f in frames for p in f["people"]]
        sc = CONTROL_DIST / (.55 * float(np.median(hs))) if hs else .02
        for f in frames:
            for p in f["people"]:
                x1, y1, x2, y2 = p["box"]
                p["xy"] = [(x1 + x2) / 2 * sc, y2 * sc]
            if f["ball"]:
                f["ball"]["xy"] = [f["ball"]["px"][0] * sc, (f["ball"]["px"][1] + 8) * sc]
    tracks = {int(k): v for k, v in data["tracks"].items()}
    names = args.names.split(",")
    dt = stride / fps
    n = len(frames)
    half = lambda t: 1 if args.halftime and t >= args.halftime else 0

    # ---- positions per track -------------------------------------------------
    pos = defaultdict(list)                     # tid -> [(frame index, t, x, y)]
    for i, f in enumerate(frames):
        for p in f["people"]:
            if "xy" in p and tracks[p["id"]]["role"] != "referee":
                pos[p["id"]].append((i, f["t"], *p["xy"]))

    # ---- which way each team attacks (per half) ------------------------------
    # goalkeepers decide it when seen, otherwise the team standing deeper defends that end
    attacks_right = {}
    for h in (0, 1):
        for k in (0, 1):
            gk = [x for tid, ps in pos.items() if tracks[tid]["team"] == k and tracks[tid]["role"] == "goalkeeper"
                  for (_, t, x, _) in ps if half(t) == h]
            if len(gk) > 10:
                attacks_right[(h, k)] = float(np.mean(gk)) < LENGTH / 2
        for k in (0, 1):
            if (h, k) not in attacks_right:
                if (h, 1 - k) in attacks_right:
                    attacks_right[(h, k)] = not attacks_right[(h, 1 - k)]
                else:
                    xs = {j: [x for tid, ps in pos.items() if tracks[tid]["team"] == j for (_, t, x, _) in ps if half(t) == h]
                          for j in (0, 1)}
                    m = {j: np.mean(v) if v else LENGTH / 2 for j, v in xs.items()}
                    attacks_right[(h, k)] = m[k] < m[1 - k]

    def norm(x, y, k, t):
        """Flip so team k always attacks left to right."""
        return (x, y) if attacks_right[(half(t), k)] else (LENGTH - x, WIDTH - y)

    # ---- running: distance, top speed, sprints -------------------------------
    players = []
    team_step = np.zeros((2, n))                # metres run per team per frame, for the live view
    for tid, ps in pos.items():
        if len(ps) < 3 or tracks[tid]["team"] < 0:
            continue
        a = np.array(ps)
        xy = a[:, 2:4].copy()
        for c in (0, 1):                         # 3 sample median removes jitter
            col = xy[:, c].copy()
            xy[1:-1, c] = np.median(np.c_[col[:-2], col[1:-1], col[2:]], axis=1)
        dist, top, sprints, in_sprint, sprint_t = 0.0, 0.0, 0, False, 0.0
        for j in range(1, len(a)):
            gap = a[j, 1] - a[j - 1, 1]
            if gap > 3 * dt:
                in_sprint = False
                continue
            step = float(np.linalg.norm(xy[j] - xy[j - 1]))
            v = step / gap
            if v > MAX_RUN_SPEED:
                continue
            dist += step
            team_step[tracks[tid]["team"], int(a[j, 0])] += step
            # speed over about 1 s is steadier than frame to frame
            back = j - max(1, round(1 / dt))
            if back >= 0 and a[j, 1] - a[back, 1] < 1.6:
                v1 = float(np.linalg.norm(xy[j] - xy[back])) / (a[j, 1] - a[back, 1])
                if v1 < MAX_RUN_SPEED:
                    top = max(top, v1)
                    if v1 > SPRINT_SPEED:
                        sprint_t = sprint_t + gap if in_sprint else gap
                        if not in_sprint:
                            in_sprint = True
                        if abs(sprint_t - 1.0) < gap / 2 + 1e-9:
                            sprints += 1
                    else:
                        in_sprint, sprint_t = False, 0.0
        k = tracks[tid]["team"]
        nxy = [norm(x, y, k, t) for (_, t, x, y) in ps]
        players.append({
            "id": tid, "team": k, "role": tracks[tid]["role"], "frames": len(ps),
            "minutes": round(len(ps) * dt / 60, 2), "distance_m": round(dist), "top_speed_kmh": round(top * 3.6, 1),
            "sprints": sprints, "avg": [round(float(np.mean([p[0] for p in nxy])), 1), round(float(np.mean([p[1] for p in nxy])), 1)],
        })

    # ---- ball and control ----------------------------------------------------
    t, ball, observed = interpolate_ball(frames, max_gap_s=1.0)
    speed = np.full(n, np.nan)
    for i in range(1, n - 1):
        if not np.isnan(ball[i - 1, 0]) and not np.isnan(ball[i + 1, 0]):
            speed[i] = np.linalg.norm(ball[i + 1] - ball[i - 1]) / (t[i + 1] - t[i - 1])

    controller = [None] * n                     # (tid, team) per frame
    for i, f in enumerate(frames):
        if np.isnan(ball[i, 0]) or (not np.isnan(speed[i]) and speed[i] > MAX_RUN_SPEED):
            continue
        best, bd = None, CONTROL_DIST
        for p in f["people"]:
            tr = tracks[p["id"]]
            if "xy" in p and tr["team"] >= 0 and tr["role"] != "referee":
                d = math.dist(p["xy"], ball[i])
                if d < bd:
                    best, bd = (p["id"], tr["team"]), d
        controller[i] = best

    # spells: consecutive frames on the ball by one player (2 frames needed to count)
    spells = []
    i = 0
    while i < n:
        c = controller[i]
        if c is None:
            i += 1
            continue
        j = i
        while j + 1 < n and (controller[j + 1] == c or (controller[j + 1] is None and j + 2 < n and controller[j + 2] == c)):
            j += 1
        if j > i or dt >= 0.25:
            spells.append({"tid": c[0], "team": c[1], "a": i, "b": j})
        i = j + 1
    # merge spells of the same player split by noise
    merged = []
    for s in spells:
        if merged and merged[-1]["tid"] == s["tid"] and t[s["a"]] - t[merged[-1]["b"]] < 1.0:
            merged[-1]["b"] = s["b"]
        else:
            merged.append(s)
    spells = merged

    # possession: a team keeps it until the other team controls the ball
    poss = [None] * n
    cur = None
    si = 0
    for i in range(n):
        while si < len(spells) and spells[si]["a"] <= i:
            cur = spells[si]["team"]
            si += 1
        poss[i] = cur
    poss_frames = Counter(p for p in poss if p is not None)
    total_poss = sum(poss_frames.values()) or 1

    # ---- passes ----------------------------------------------------------------
    passes, turnovers = [], []
    for s1, s2 in zip(spells, spells[1:]):
        gap = t[s2["a"]] - t[s1["b"]]
        if gap > 6 or s1["tid"] == s2["tid"]:
            continue
        a, b = ball[s1["b"]], ball[s2["a"]]
        k = s1["team"]
        ax, ay = norm(*a, k, t[s1["b"]])
        bx, by = norm(*b, k, t[s1["b"]])
        length = math.dist((ax, ay), (bx, by))
        item = {"team": k, "from": s1["tid"], "to": s2["tid"], "t": round(float(t[s1["b"]]), 2),
                "x1": round(ax, 1), "y1": round(ay, 1), "x2": round(bx, 1), "y2": round(by, 1), "len": round(length, 1),
                "raw": [round(float(a[0]), 1), round(float(a[1]), 1), round(float(b[0]), 1), round(float(b[1]), 1)],
                "t2": round(float(t[s2["a"]]), 2)}
        if s2["team"] == k:
            item["ok"] = True
            item["progressive"] = not IMAGE and (bx - ax) >= 10 and bx > LENGTH * 0.4
            item["final_third"] = not IMAGE and bx >= LENGTH * 2 / 3 > ax
            passes.append(item)
        elif length > 4:
            item["ok"] = False
            item["progressive"] = item["final_third"] = False
            passes.append(item)
        else:
            turnovers.append({"won_by": s2["team"], "t": item["t"], "x": round(bx, 1), "y": round(by, 1)})

    # ---- shots -------------------------------------------------------------------
    shots, last_shot_t = [], -99
    for i in (range(1, n - 1) if not IMAGE else []):
        if np.isnan(speed[i]) or speed[i] < SHOT_SPEED or t[i] - last_shot_t < 2:
            continue
        k = poss[i]
        if k is None:
            continue
        right = attacks_right[(half(t[i]), k)]
        gx = LENGTH if right else 0.0
        v = (ball[i + 1] - ball[i - 1])
        if (v[0] > 0) != right or abs(v[0]) < 1e-6:
            continue
        x0, y0 = ball[i - 1]
        if abs(gx - x0) > SHOT_RANGE:
            continue
        cross_y = y0 + v[1] * (gx - x0) / v[0]
        if abs(cross_y - WIDTH / 2) > GOAL_W / 2 + 8:     # heading nowhere near the goal
            continue
        # origin: where the shooter last had it, if that was just before
        start = i - 1
        prev = [s for s in spells if s["b"] <= i and s["team"] == k]
        if prev and t[i] - t[prev[-1]["b"]] < 1.5:
            start = prev[-1]["b"]
        sx, sy = ball[start]
        on_target = abs(cross_y - WIDTH / 2) <= GOAL_W / 2 + 0.5
        # goal: ball later seen past the line between the posts
        goal = False
        for j in range(i, min(n, i + int(3 / dt))):
            if observed[j] and ((ball[j, 0] >= LENGTH) if right else (ball[j, 0] <= 0)) and abs(ball[j, 1] - WIDTH / 2) < GOAL_W / 2:
                goal = True
                break
        nx, ny = norm(sx, sy, k, t[i])
        shooter = prev[-1]["tid"] if prev else None
        shots.append({"team": k, "t": round(float(t[i]), 2), "x": round(nx, 1), "y": round(ny, 1),
                      "speed_kmh": round(float(speed[i]) * 3.6), "on_target": bool(on_target), "goal": goal,
                      "xg": round(xg(LENGTH - nx, ny - WIDTH / 2), 3), "player": shooter,
                      "end_y": round(float(cross_y if right else WIDTH - cross_y), 1)})
        last_shot_t = t[i]

    # ---- heatmaps and territory ------------------------------------------------
    def heat(points):
        h, _, _ = np.histogram2d([p[0] for p in points], [p[1] for p in points], bins=GRID,
                                 range=[[0, LENGTH], [0, WIDTH]])
        m = h.max() or 1
        return (h / m).round(3).T.tolist()       # rows = y, cols = x

    team_pts = {0: [], 1: []}
    for tid, ps in pos.items():
        k = tracks[tid]["team"]
        if k in team_pts and tracks[tid]["role"] == "player":
            team_pts[k] += [norm(x, y, k, tt) for (_, tt, x, y) in ps]
    ball_pts_team0 = [norm(*ball[i], 0, t[i]) for i in range(n) if not np.isnan(ball[i, 0])]
    thirds = Counter(min(2, int(x // (LENGTH / 3))) for x, _ in ball_pts_team0)
    tsum = sum(thirds.values()) or 1

    # ---- pass network: the 11 most seen tracks per team --------------------------
    network = {}
    for k in (0, 1):
        top = sorted([p for p in players if p["team"] == k], key=lambda p: -p["frames"])[:11]
        ids = {p["id"] for p in top}
        edges = Counter((min(p["from"], p["to"]), max(p["from"], p["to"])) for p in passes
                        if p["team"] == k and p["ok"] and p["from"] in ids and p["to"] in ids)
        touches = Counter(p["from"] for p in passes if p["team"] == k)
        network[k] = {"nodes": [{"id": p["id"], "x": p["avg"][0], "y": p["avg"][1], "passes": touches[p["id"]],
                                 "role": p["role"]} for p in top],
                      "edges": [{"a": a, "b": b, "n": c} for (a, b), c in edges.items()]}

    # ---- possession timeline ---------------------------------------------------
    buckets = 20
    timeline = []
    for b in range(buckets):
        seg = [p for p in poss[b * n // buckets:(b + 1) * n // buckets] if p is not None]
        timeline.append(round(sum(1 for p in seg if p == 0) / len(seg), 3) if seg else None)

    # ---- team summary ----------------------------------------------------------
    teams = []
    for k in (0, 1):
        tp = [p for p in passes if p["team"] == k]
        ts = [s for s in shots if s["team"] == k]
        tpl = [p for p in players if p["team"] == k]
        fastest = max(tpl, key=lambda p: p["top_speed_kmh"], default=None)
        teams.append({
            "name": names[k], "colour": data["team_colours"][str(k)] if str(k) in data["team_colours"] else data["team_colours"][k],
            "possession": round(poss_frames[k] / total_poss * 100, 1),
            "passes": len(tp), "completed": sum(p["ok"] for p in tp),
            "accuracy": round(sum(p["ok"] for p in tp) / len(tp) * 100, 1) if tp else 0,
            "progressive": sum(p["progressive"] for p in tp), "final_third": sum(p["final_third"] for p in tp),
            "shots": len(ts), "on_target": sum(s["on_target"] for s in ts), "goals": sum(s["goal"] for s in ts),
            "xg": round(sum(s["xg"] for s in ts), 2), "turnovers_won": sum(1 for x in turnovers if x["won_by"] == k),
            "distance_km": round(sum(p["distance_m"] for p in tpl) / 1000, 2),
            "top_speed_kmh": fastest["top_speed_kmh"] if fastest else 0, "fastest_id": fastest["id"] if fastest else None,
            "sprints": sum(p["sprints"] for p in tpl),
        })

    report = {
        "meta": {
            "video": Path(data["video"]).name, "fps": fps, "stride": stride, "frames": n,
            "duration_s": round(t[-1] - t[0] + dt, 1) if n else 0,
            "pitch_mapped": round(sum(f["h"] for f in frames) / max(1, n) * 100, 1),
            "ball_seen": round(float(observed.sum()) / max(1, n) * 100, 1),
            "processing_s": data["seconds_processed"], "tracks": len(players),
            "pitch": [LENGTH, WIDTH], "grid": GRID,
        },
        "teams": teams, "players": sorted(players, key=lambda p: -p["distance_m"]),
        "passes": passes, "shots": shots, "turnovers": turnovers,
        "heat": {"team0": heat(team_pts[0]), "team1": heat(team_pts[1]), "ball": heat(ball_pts_team0)},
        "territory": [round(thirds[i] / tsum * 100, 1) for i in range(3)],
        "network": network, "timeline": timeline,
        "live": {
            "t": [round(float(x), 3) for x in t],
            "poss": poss,
            "ball": [None if np.isnan(b[0]) else [round(float(b[0]), 2), round(float(b[1]), 2)] for b in ball],
            "dist": [[round(float(v)) for v in np.cumsum(team_step[k])] for k in (0, 1)],
            "attacks_right": [attacks_right[(0, 0)], attacks_right[(0, 1)]],
        },
    }
    (run / "stats.json").write_text(json.dumps(report, indent=1, default=lambda o: o.item()))
    for tm in teams:
        print(f"{tm['name']}: possession {tm['possession']}%, passes {tm['completed']}/{tm['passes']}, "
              f"shots {tm['shots']} ({tm['on_target']} on target, xG {tm['xg']}), distance {tm['distance_km']} km")
    print(f"wrote {run / 'stats.json'}")


if __name__ == "__main__":
    main()

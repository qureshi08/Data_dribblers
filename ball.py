"""Picks the match ball out of every ball-shaped thing the detector saw.

Broadcasts are full of other balls: spare balls by the touchline, balls in the ball boys' hands, warm up balls. Taking
the single most confident detection per frame (what analyze.py did) locks onto a sharp, still spare ball and never
follows the small, blurred match ball (Barcelona v Athletic, 2026-11-02: 195 of 366 frames on one spot by the halfway
touchline). Here every candidate in every frame is mapped onto the pitch and the most believable single path is chosen:

  * off the pitch (more than 1 m outside the lines) is never the ball in play;
  * a candidate that sits still for over 2 s within 4 m of a line is a spare ball;
  * between frames the ball cannot move faster than about 40 m/s;
  * the ball in play is usually near players;
  * a frame may have no ball (out of shot, hidden, too blurred), which costs a little but beats a wrong ball.

Candidates come from analyze.py (rec["balls"]) when present; older runs are re-detected with the ball model.
Each frame's pitch mapping is fitted from the players' own foot positions (pixel -> metres pairs in tracks.json),
so this needs no pitch keypoints of its own.

    python ball.py runs/match
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
MAX_SPEED, MISS, NEAR_W = 40.0, 2.2, 0.06


def frame_h(rec):
    """pixel -> pitch homography: the one analyze.py fitted to the pitch markings (rec["H"]) when saved, since it holds
    at the touchlines; otherwise one fitted to the players' feet in this frame (weak at the frame edges), or None"""
    if rec.get("H"):
        return np.array(rec["H"], dtype=float).reshape(3, 3)
    src, dst = [], []
    for p in rec["people"]:
        if "xy" in p:
            x1, y1, x2, y2 = p["box"]; src.append([(x1 + x2) / 2, y2]); dst.append(p["xy"])
    if len(src) < 6:
        return None
    H, _ = cv2.findHomography(np.float32(src), np.float32(dst), cv2.RANSAC, 1.5)
    return H


def to_xy(H, px):
    v = H @ np.array([px[0], px[1], 1.0]); return [float(v[0] / v[2]), float(v[1] / v[2])]


def detect(run, frames, video, imgsz=1280):
    """ball candidates for every analysed frame with the dedicated ball model (older runs)"""
    from ultralytics import YOLO
    model = YOLO(HERE / "data" / "football-ball-detection.pt")
    from frames import frames_at
    by_f = {rec["f"]: rec for rec in frames}
    for rec in frames:
        rec["balls"] = []
    for k, (f, frame) in enumerate(frames_at(video, by_f)):   # one pass through the video, never a seek per frame
        r = model.predict(frame, imgsz=imgsz, conf=0.1, verbose=False)[0]
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            by_f[f]["balls"].append({"px": [round(float((x1 + x2) / 2), 1), round(float((y1 + y2) / 2), 1)], "conf": round(float(c), 3)})
        if k % 25 == 0:
            print(f"  ball candidates: {k + 1}/{len(frames)} frames", flush=True)


def tiles(w, h, cols=3, rows=2, overlap=.2):
    """overlapping tiles covering the frame, as (x, y, w, h)"""
    tw, th = int(w / (cols - (cols - 1) * overlap)), int(h / (rows - (rows - 1) * overlap))
    xs = np.linspace(0, w - tw, cols).astype(int); ys = np.linspace(0, h - th, rows).astype(int)
    return [(x, y, tw, th) for y in ys for x in xs]


def sliced(model, frame, imgsz=1280, conf=0.1):
    """ball candidates from overlapping tiles, each enlarged to imgsz (sliced inference, as SAHI does).

    On a wide tactical camera the ball is 4 to 8 pixels across, smaller than anything the model learned from the
    Bundesliga broadcasts it was trained on; a 3 x 2 tile enlarged to the model's size makes it about twice as big."""
    h, w = frame.shape[:2]
    crops = [frame[y:y + th, x:x + tw] for x, y, tw, th in tiles(w, h)]
    out = []
    for (x, y, _, _), r in zip(tiles(w, h), model.predict(crops, imgsz=imgsz, conf=conf, verbose=False)):
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            out.append({"px": [round(float((x1 + x2) / 2 + x), 1), round(float((y1 + y2) / 2 + y), 1)], "conf": round(float(c), 3)})
    return out


def merge(a, b, r=6):
    """candidates from two passes, keeping the more confident of any two closer than r pixels"""
    out = []
    for c in sorted(a + b, key=lambda c: -c["conf"]):
        if all(np.hypot(c["px"][0] - o["px"][0], c["px"][1] - o["px"][1]) > r for o in out):
            out.append(c)
    return out


def add_sliced(frames, video, imgsz=1280):
    """adds sliced ball candidates to every analysed frame (one pass through the video)"""
    from ultralytics import YOLO
    model = YOLO(HERE / "data" / "football-ball-detection.pt")
    from frames import frames_at
    by_f = {rec["f"]: rec for rec in frames}
    for k, (f, frame) in enumerate(frames_at(video, by_f)):
        by_f[f]["balls"] = merge(by_f[f].get("balls", []), sliced(model, frame, imgsz))
        by_f[f]["sliced"] = True
        if k % 10 == 0:
            print(f"  sliced ball search: {k + 1}/{len(frames)} frames", flush=True)


def choose(frames, roles=None):
    """Short ball tracks first, then the best chain of them.

    1. candidates on the pitch are linked frame to frame into short tracks (each step within ball speed, gaps of up
       to 1 s allowed);
    2. a track that stays put (under 1.5 m of movement) for over 2 s with no player close by is not the match ball:
       spare balls by the lines, the painted penalty spot (Barcelona v Athletic: 13 s at 11 m, 32 m);
    3. the chosen tracks never overlap in time and every jump between them is physically possible, so a frame with
       no ball no longer lets the path teleport (the single frame version jumped 67 m)."""
    t = [rec["t"] for rec in frames]
    pts = []   # (frame, candidate)
    for i, rec in enumerate(frames):
        H = frame_h(rec)
        for b in rec.get("balls", []):
            if H is None:
                continue
            xy = to_xy(H, b["px"])
            if -.5 <= xy[0] <= 105.5 and -.5 <= xy[1] <= 68.5:   # half a metre over the lines at most (the mapping stretches at the frame edge)
                near = min((np.hypot(p["xy"][0] - xy[0], p["xy"][1] - xy[1]) for p in rec["people"]
                            if "xy" in p and (roles or {}).get(str(p["id"]), "player") in ("player", "goalkeeper")), default=30)   # officials never count: a linesman stands by the spare balls
                pts.append((i, {**b, "xy": xy, "near": near}))
    # fixtures: a spot (3 m cells) that holds a "ball" in over 30% of all frames is not the match ball. A real ball
    # never stays in one place for a third of the clip; spare balls, painted spots and line marks do. Mapping near
    # the frame edge wobbles by metres, so cells are checked with their neighbours.
    cell = lambda xy: (int(xy[0] // 3), int(xy[1] // 3))
    seen = {}
    for i, c in pts:
        seen.setdefault(cell(c["xy"]), set()).add(i)
    def hits(cx, cy):
        return set().union(*(seen.get((cx + dx, cy + dy), set()) for dx in (-1, 0, 1) for dy in (-1, 0, 1)))
    span = frames[-1]["t"] - frames[0]["t"]
    share = .3 if span > 10 else .6   # in a short clip a dribbled ball stays in one area for a third of it; a fixture is there almost always
    fixed = {k for k in seen if len(hits(*k)) > share * len(frames)}
    pts = [(i, c) for i, c in pts if cell(c["xy"]) not in fixed]
    # 1. link into tracks
    tracks = []
    for i, c in sorted(pts, key=lambda q: q[0]):
        best, bd = None, 1e9
        for tr in tracks:
            j, last = tr[-1]
            dt = t[i] - t[j]
            if j >= i or dt > 1.0:
                continue
            d = np.hypot(last["xy"][0] - c["xy"][0], last["xy"][1] - c["xy"][1])
            if d <= MAX_SPEED * dt + 1.0 and d < bd:
                best, bd = tr, d
        if best is not None and best[-1][0] < i:
            best.append((i, c))
        else:
            tracks.append([(i, c)])
    # 2. score, dropping still decoys
    scored = []
    for tr in tracks:
        xs = np.array([c["xy"] for _, c in tr]); dur = t[tr[-1][0]] - t[tr[0][0]]
        extent = float(np.hypot(*(xs.max(0) - xs.min(0))))
        near = np.median([c["near"] for _, c in tr])
        settled = np.mean(np.hypot(*(xs - np.median(xs, 0)).T) < 3.0)   # share of the track within 3 m of its usual spot
        if dur > 2.0 and settled > .8 and near > 1.5:
            continue
        score = sum(c["conf"] - 0.25 - 0.06 * min(c["near"], 25) for _, c in tr) + 0.15 * min(extent, 20)   # the ball in play is near players
        if score > 0:
            scored.append((tr[0][0], tr[-1][0], score, tr))
    # 3. best chain of non overlapping, physically possible tracks (weighted interval scheduling)
    scored.sort(key=lambda q: q[1])
    best, prev = [], []
    for k, (s0, e0, sc, tr) in enumerate(scored):
        b0, p0 = sc, -1
        for m in range(k):
            s1, e1, _, tr1 = scored[m]
            if e1 >= s0:
                continue
            gap = max(t[s0] - t[e1], 1e-3); a, c = tr1[-1][1]["xy"], tr[0][1]["xy"]
            if np.hypot(a[0] - c[0], a[1] - c[1]) > MAX_SPEED * gap + 2:
                continue
            if best[m] + sc > b0:
                b0, p0 = best[m] + sc, m
        best.append(b0); prev.append(p0)
    path = [None] * len(frames)
    if scored:
        k = int(np.argmax(best))
        while k >= 0:
            for i, c in scored[k][3]:
                path[i] = c
            k = prev[k]
    return path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--slice", action="store_true", help="also search 3 x 2 enlarged tiles for a small ball (about 6 times the ball model's time)")
    a = ap.parse_args()
    path = Path(a.run) / "tracks.json"; data = json.loads(path.read_text())
    frames = data["frames"]
    if not any("balls" in r for r in frames):
        detect(a.run, frames, data["video"], a.imgsz)
    if a.slice and not all(r.get("sliced") for r in frames):
        add_sliced(frames, data["video"], a.imgsz)
    chosen = choose(frames, {k: v["role"] for k, v in data["tracks"].items()})
    for rec, c in zip(frames, chosen):
        rec["ball"] = None if c is None else {"px": c["px"], "xy": [round(c["xy"][0], 2), round(c["xy"][1], 2)]}
    # short gaps (up to 0.6 s) between two sightings are filled in a straight line, so the ball does not blink out
    seen = [i for i, c in enumerate(chosen) if c is not None]
    for a, b in zip(seen, seen[1:]):
        ta, tb = frames[a]["t"], frames[b]["t"]
        if b - a < 2 or tb - ta > .6:
            continue
        A, B = frames[a]["ball"], frames[b]["ball"]
        for i in range(a + 1, b):
            u = (frames[i]["t"] - ta) / (tb - ta)
            frames[i]["ball"] = {"px": [round(A["px"][k] + (B["px"][k] - A["px"][k]) * u, 1) for k in (0, 1)],
                                 "xy": [round(A["xy"][k] + (B["xy"][k] - A["xy"][k]) * u, 2) for k in (0, 1)], "filled": True}
    path.write_text(json.dumps(data))
    seen = sum(c is not None for c in chosen); cand = sum(len(r.get("balls", [])) for r in frames)
    print(f"ball: {cand} candidates in {len(frames)} frames, match ball chosen in {seen} ({seen / max(1, len(frames)):.0%})")


if __name__ == "__main__":
    main()

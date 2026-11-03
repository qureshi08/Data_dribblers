"""Split players into two teams by shirt colour.

The grass is measured in every frame (its own hue, saturation and brightness) and only pixels close
to it are dropped, so green shirts survive. A fixed "green = grass" range deleted a whole neon green
team on clip 121364 (2026-11-02).

Runs at the end of analyze.py; also fixes an existing run without re-running detection:

    python teams.py runs/match
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans

from stitch import stitch

MAX_SAMPLES = 40


def grass_hsv(frame):
    """Median colour of the pitch: green-ish pixels in the lower two thirds of the frame."""
    small = cv2.resize(frame, None, fx=.125, fy=.125, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[small.shape[0] // 3:].reshape(-1, 3).astype(int)
    g = hsv[(hsv[:, 0] >= 25) & (hsv[:, 0] <= 95) & (hsv[:, 1] > 40)]
    return np.median(g, axis=0) if len(g) > 50 else np.array([60, 120, 120])


def torso(frame, box, grass):
    """(shirt colour histogram, mean BGR, shirt pixels in Lab) with grass, deep shadow and glare removed.

    The histogram is over Lab a and b only (colour without brightness): the same shirt in sun and in shade lands in
    the same bins. Splitting on full HSV histograms put sunlit and shaded players in different "teams"
    (Barcelona v Athletic, 2026-11-02: 56 tracks against 25, team colours grey and brown)."""
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = y2 - y1, x2 - x1
    if h < 20 or w < 8:
        return None, None, None
    crop = frame[max(y1 + int(h * .15), 0):y1 + int(h * .5), max(x1 + int(w * .2), 0):x2 - int(w * .2)]
    if crop.size == 0:
        return None, None, None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    d = hsv.astype(int) - grass
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    keep = ~((np.abs(d[..., 0]) < 9) & (np.abs(d[..., 1]) < 70) & (np.abs(d[..., 2]) < 55)) & (lab[..., 0] > 30) & (lab[..., 0] < 250)
    if keep.sum() < 15:
        return None, None, None
    hist = cv2.calcHist([lab], [1, 2], keep.astype(np.uint8), [10, 10], [78, 178, 78, 178]).ravel()
    return np.sqrt(hist / max(hist.sum(), 1)), crop[keep].mean(axis=0), lab[keep].astype(float) * LAB_W


LAB_W = np.array([.35, 1, 1])   # brightness counts a little: enough for black v white kits, not enough for sun v shade


def hex_ab(h):
    """'#rrggbb' -> weighted Lab in OpenCV's 8 bit scale"""
    h = h.lstrip('#'); bgr = np.uint8([[[int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)]]])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(float) * LAB_W


def kit_colour(pix):
    """A kit's display colour from its shirt pixels: the typical colour of its strongly coloured pixels, or white or
    black when the kit has little colour. Averaging every pixel turned blaugrana stripes into brown and a white kit
    in shade into grey, and both then showed as white on the page."""
    if not pix:
        return "#c8c8c8"
    P = np.vstack(pix) / LAB_W                          # back to plain 8 bit Lab
    chroma = np.hypot(P[:, 1] - 128, P[:, 2] - 128)
    strong = P[chroma > 18]
    if len(strong) < .25 * len(P):                      # a white, grey or black kit
        return "#f2f2f2" if np.median(P[:, 0]) > 110 else "#1b1b1b"
    lab = np.median(strong, axis=0); lab[0] = max(lab[0], 120)
    b, g, r = cv2.cvtColor(np.uint8([[lab]]), cv2.COLOR_LAB2BGR)[0, 0]
    return "#%02x%02x%02x" % (r, g, b)


def assign(video, frames, tracks, hints=None):
    """Sets tracks[id]["team"] for players and goalkeepers; returns {team: '#rrggbb'}.

    hints: the two kit colours from the upload form ('#rrggbb', home first). With hints every shirt pixel votes for
    the nearer kit colour (by colour, not brightness), which also handles striped kits. Without hints the shirts are
    grouped into three: the two biggest groups are the teams, a small third group that matches neither kit is the
    officials, whom the detector sometimes calls players."""
    features, colours, pixels = defaultdict(list), defaultdict(list), defaultdict(list)
    players = {int(k) for k, v in tracks.items() if v["role"] in ("player", "goalkeeper")}
    from frames import frames_at
    by_f = {rec["f"]: rec for rec in frames}
    for f, frame in frames_at(video, by_f):   # one pass through the video, never a seek per frame
        rec = by_f[f]
        need = [p for p in rec["people"] if p["id"] in players and len(features[p["id"]]) < MAX_SAMPLES]
        if not need:
            continue
        grass = grass_hsv(frame)
        for p in need:
            f, c, px = torso(frame, p["box"], grass)
            if f is not None:
                features[p["id"]].append(f); colours[p["id"]].append(c)
                pixels[p["id"]].append(px[np.random.default_rng(p["id"]).choice(len(px), min(len(px), 60), replace=False)])

    tids = [t for t in features if features[t] and tracks[str(t)]["role"] == "player"]
    for k, v in tracks.items():
        if v["role"] in ("player", "goalkeeper"):
            v["team"] = -1
    # each player's typical shirt colour: median Lab with brightness at half weight. Broadcast compression and light
    # move kit colours a long way from the real kit (Barcelona's garnet and blue stripes read as a dull dark grey,
    # Athletic's lavender stays lavender), so players are grouped by what the camera shows, not matched to swatches.
    med = {t: np.median(np.vstack(pixels[t]) / LAB_W, axis=0) for t in tids}
    X = np.array([[.5 * med[t][0], med[t][1], med[t][2]] for t in tids]); wts = np.array([len(features[t]) for t in tids])
    # three groups: the two biggest are the teams; a clearly separate third is officials the detector called players
    # (in two groups the neon referees were the biggest difference and swallowed a whole "team")
    k3 = 3 if len(tids) >= 6 else 2
    km3 = KMeans(n_clusters=k3, n_init=10, random_state=0).fit(X, sample_weight=wts)
    size = np.bincount(km3.labels_, weights=wts, minlength=k3); big = list(np.argsort(-size)[:2])
    labels, centers = km3.labels_.copy(), km3.cluster_centers_
    if k3 == 3:
        other = [q for q in range(3) if q not in big][0]
        gap = np.linalg.norm(centers[big[0]] - centers[big[1]])
        near = big[int(np.argmin([np.linalg.norm(centers[other] - centers[q]) for q in big]))]
        if np.linalg.norm(centers[other] - centers[near]) > .5 * gap:
            for t, lab in zip(tids, labels):
                if lab == other:
                    tracks[str(t)]["role"] = "referee"
        else:
            labels[labels == other] = near
    keep = [i for i, lab in enumerate(labels) if lab in big]
    tids = [tids[i] for i in keep]; labels = np.array([big.index(labels[i]) for i in keep])
    km = type("KM", (), {})(); km.labels_ = labels; km.cluster_centers_ = centers[big]
    order = [0, 1]
    if hints and len(hints) == 2:
        # the kit colours from the upload form only name the groups: the pairing with the smaller total distance wins
        H = []
        for h in hints:
            lab = cv2.cvtColor(np.uint8([[[int(h.split('+')[0].lstrip('#')[i:i + 2], 16) for i in (4, 2, 0)]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
            H.append([.5 * lab[0], lab[1], lab[2]])
        c = km.cluster_centers_; dist = lambda i, j: np.linalg.norm(c[i] - H[j])
        order = [0, 1] if dist(0, 0) + dist(1, 1) <= dist(0, 1) + dist(1, 0) else [1, 0]   # order[cluster] = team
    for t, lab in zip(tids, km.labels_):
        tracks[str(t)]["team"] = int(order[lab])
    for t in med:   # each player's typical shirt colour, so stitch.py can re-join a lost player by appearance
        tracks[str(t)]["lab"] = [round(float(v), 1) for v in med[t]]
    if hints and len(hints) == 2:
        team_colour = {k: hints[k].split('+')[0] for k in (0, 1)}
    else:
        team_colour = {k: kit_colour([px for t in tids if tracks[str(t)]["team"] == k for px in pixels[t]]) for k in (0, 1)}

    # people who stay off the pitch (camera crew, ball boys, staff) are never players: no passing lanes, no stats
    for k, v in tracks.items():
        if v["role"] not in ("player", "goalkeeper"):
            continue
        pts = [p["xy"] for rec in frames for p in rec["people"] if str(p["id"]) == k and "xy" in p]
        if len(pts) >= 3 and np.mean([(x < -1.5 or x > 106.5 or y < -1.5 or y > 69.5) for x, y in pts]) > .6:
            v["role"], v["team"] = "referee", -1

    # a keeper in a bright kit is sometimes classed as a referee: an "official" who spends most of the clip inside a
    # penalty area is the goalkeeper (referees and linesmen roam the pitch and the touchlines)
    # (the keeper can stand well outside the box while his team attacks, so "deepest person on the pitch" is the test,
    # not "inside the box"; linesmen run the touchlines and are skipped)
    for k, v in tracks.items():
        if v["role"] != "referee":
            continue
        deepest, ys = [], []
        for rec in frames:
            me = next((p for p in rec["people"] if str(p["id"]) == k and "xy" in p), None)
            xs = [p["xy"][0] for p in rec["people"] if "xy" in p and str(p["id"]) != k]
            if me is None or len(xs) < 6:
                continue
            ys.append(me["xy"][1]); deepest.append(me["xy"][0] <= min(xs) + 1 or me["xy"][0] >= max(xs) - 1)
        if len(deepest) >= 5 and np.mean(deepest) > .6 and 8 < np.mean(ys) < 60:
            v["role"] = "goalkeeper"

    # goalkeepers join the team whose outfield players stand closest on average
    for k, v in tracks.items():
        if v["role"] != "goalkeeper":
            continue
        votes = []
        for rec in frames:
            gk = next((p for p in rec["people"] if str(p["id"]) == k and "xy" in p), None)
            if gk is None:
                continue
            cents = []
            for team in (0, 1):
                pts = [p["xy"] for p in rec["people"] if "xy" in p and tracks[str(p["id"])]["role"] == "player"
                       and tracks[str(p["id"])]["team"] == team]
                cents.append(np.mean(pts, axis=0) if pts else None)
            if all(c is not None for c in cents):
                votes.append(int(np.argmin([np.linalg.norm(np.array(gk["xy"]) - c) for c in cents])))
        if votes:
            v["team"] = Counter(votes).most_common(1)[0][0]
    return team_colour


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--colours", default="", help="home and away kit colours, e.g. '#ffffff,#a50044'")
    args = ap.parse_args()
    path = Path(args.run) / "tracks.json"
    data = json.loads(path.read_text())
    data["team_colours"] = assign(data["video"], data["frames"], data["tracks"], [c for c in args.colours.split(",") if c] or None)
    before = len(data["tracks"])
    data["tracks"] = stitch(data["frames"], data["tracks"])
    path.write_text(json.dumps(data))
    roles = Counter((v["role"], v["team"]) for v in data["tracks"].values())
    print(f"teams {data['team_colours']}, tracks {before} -> {len(data['tracks'])}, {dict(roles)}")


if __name__ == "__main__":
    main()

"""Join broken tracks. When a player's track ends and a new track of the same team starts
shortly after, close by on the pitch, they are the same player.

Runs at the end of analyze.py; can also be run on an existing run:

    python stitch.py runs/match
"""
import argparse
import json
import math
from pathlib import Path

MAX_GAP_S = 5.0               # a player lost for up to 5 s (camera pan, crossing) can be re-joined...
BASE_M, RUN_MS = 3.0, 7.5      # ...if the new track starts within 3 m + 7.5 m/s x gap of where they were lost
LOOK = 22.0                    # ...and their shirts look alike (median Lab, brightness at half weight; teams.py)


def stitch(frames, tracks):
    """Rewrites person ids in frames in place and returns the merged tracks dict."""
    span = {}
    for i, f in enumerate(frames):
        for p in f["people"]:
            if "xy" not in p:
                continue
            s = span.setdefault(p["id"], {"a": i, "b": i, "first": p["xy"], "last": p["xy"]})
            s["b"], s["last"] = i, p["xy"]

    t = [f["t"] for f in frames]
    chains = []                  # open chains: root id, team, role, end frame, last xy
    root = {}
    for tid in sorted(span, key=lambda k: span[k]["a"]):
        tr, s = tracks[str(tid)], span[tid]
        best, bd = None, None
        if tr["role"] != "referee" and tr["team"] >= 0:
            for c in chains:
                if c["team"] != tr["team"] or c["role"] != tr["role"] or c["b"] >= s["a"]:
                    continue
                gap = t[s["a"]] - t[c["b"]]
                if gap > MAX_GAP_S:
                    continue
                d = math.dist(c["last"], s["first"])
                if d > BASE_M + RUN_MS * gap:
                    continue
                look = 0.0
                if c.get("lab") and tr.get("lab"):
                    a, b = c["lab"], tr["lab"]
                    look = math.dist([.5 * a[0], a[1], a[2]], [.5 * b[0], b[1], b[2]])
                    if look > LOOK:
                        continue        # same team, close by, but a different looking shirt: a different player
                score = d / (BASE_M + RUN_MS * gap) + look / LOOK
                if bd is None or score < bd:
                    best, bd = c, score
        if best is None:
            chains.append({"root": tid, "team": tr["team"], "role": tr["role"], "b": s["b"], "last": s["last"], "lab": tr.get("lab")})
            root[tid] = tid
        else:
            best["b"], best["last"] = s["b"], s["last"]
            root[tid] = best["root"]

    for f in frames:
        for p in f["people"]:
            p["id"] = root.get(p["id"], p["id"])
    return {k: v for k, v in tracks.items() if root.get(int(k), int(k)) == int(k)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    args = ap.parse_args()
    path = Path(args.run) / "tracks.json"
    data = json.loads(path.read_text())
    before = len(data["tracks"])
    data["tracks"] = stitch(data["frames"], data["tracks"])
    path.write_text(json.dumps(data))
    print(f"tracks: {before} -> {len(data['tracks'])}")


if __name__ == "__main__":
    main()

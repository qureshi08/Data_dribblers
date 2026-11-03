"""Contact sheet for checking the ball by eye: one tile per analysed frame, zoomed on the chosen ball (green ring,
orange when gap filled) or the full frame with "no ball" when none was chosen.

    python ball_sheet.py runs/match out.jpg [--every 2]
"""
import argparse
import json

import cv2
import numpy as np

from frames import frames_at


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("out"); ap.add_argument("--every", type=int, default=2)
    a = ap.parse_args()
    d = json.load(open(f"{a.run}/tracks.json"))
    frames = d["frames"][::a.every]
    by_f = {r["f"]: r for r in frames}
    tiles = []
    for f, fr in frames_at(d["video"], by_f):
        b = by_f[f].get("ball"); h, w = fr.shape[:2]
        if b:
            x, y = (int(v) for v in b["px"])
            cv2.circle(fr, (x, y), 14, (0, 165, 255) if b.get("filled") else (60, 220, 60), 2)
            x0, y0 = min(max(x - 160, 0), w - 320), min(max(y - 90, 0), h - 180)
            tile = cv2.resize(fr[y0:y0 + 180, x0:x0 + 320], (320, 180))
        else:
            tile = cv2.resize(fr, (320, 180)); cv2.putText(tile, "no ball", (8, 170), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
        cv2.putText(tile, f"{by_f[f]['t']:.1f}s", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
        tiles.append(tile)
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(a.out, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {a.out}: {len(frames)} frames")


if __name__ == "__main__":
    main()

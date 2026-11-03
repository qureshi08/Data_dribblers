"""Step 3: annotated video with team colours, the ball and a live radar of the pitch.

    python render.py runs/match
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pitch import LENGTH, PITCH_VERTICES, WIDTH

EDGES = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8), (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
         (16, 17), (18, 19), (19, 20), (20, 21), (23, 24), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
         (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17), (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30)]
REF, BALL = (0, 200, 255), (0, 79, 217)      # BGR: yellow referee, brand orange ball


def hex_bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def radar_base(scale):
    w, h, pad = int(LENGTH * scale), int(WIDTH * scale), 12
    img = np.full((h + 2 * pad, w + 2 * pad, 3), (79, 122, 45), np.uint8)   # brand green
    pts = (PITCH_VERTICES * scale + pad).astype(int)
    for a, b in EDGES:
        cv2.line(img, tuple(pts[a - 1]), tuple(pts[b - 1]), (250, 254, 255), 1, cv2.LINE_AA)
    cv2.circle(img, (pad + w // 2, pad + h // 2), int(9.15 * scale), (250, 254, 255), 1, cv2.LINE_AA)
    return img, pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    args = ap.parse_args()
    run = Path(args.run)
    data = json.loads((run / "tracks.json").read_text())
    tracks, frames = data["tracks"], data["frames"]
    colours = {int(k): hex_bgr(v) for k, v in data["team_colours"].items()}
    cap = cv2.VideoCapture(data["video"])
    W, H = data["width"], data["height"]
    out_fps = data["fps"] / data["stride"]
    writer = cv2.VideoWriter(str(run / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))
    scale = W * 0.24 / LENGTH
    base, pad = radar_base(scale)
    snap_at = frames[len(frames) // 2]["f"] if frames else -1

    for rec in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, rec["f"])
        ok, img = cap.read()
        if not ok:
            break
        radar = base.copy()
        for p in rec["people"]:
            tr = tracks[str(p["id"])]
            col = REF if tr["role"] == "referee" else colours.get(tr["team"], (200, 200, 200))
            x1, y1, x2, y2 = map(int, p["box"])
            cx = (x1 + x2) // 2
            cv2.ellipse(img, (cx, y2), (max(8, (x2 - x1) // 2), 6), 0, -45, 235, col, 2, cv2.LINE_AA)
            if tr["role"] != "referee":
                cv2.putText(img, str(p["id"]), (cx - 8, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            if "xy" in p:
                c = (int(p["xy"][0] * scale + pad), int(p["xy"][1] * scale + pad))
                cv2.circle(radar, c, 5, col, -1, cv2.LINE_AA)
                cv2.circle(radar, c, 5, (26, 18, 5), 1, cv2.LINE_AA)
        if rec["ball"]:
            bx, by = map(int, rec["ball"]["px"])
            cv2.drawMarker(img, (bx, by - 14), BALL, cv2.MARKER_TRIANGLE_DOWN, 14, 2, cv2.LINE_AA)
            if "xy" in rec["ball"]:
                c = (int(rec["ball"]["xy"][0] * scale + pad), int(rec["ball"]["xy"][1] * scale + pad))
                cv2.circle(radar, c, 4, (255, 254, 250), -1, cv2.LINE_AA)
                cv2.circle(radar, c, 4, BALL, 1, cv2.LINE_AA)
        rh, rw = radar.shape[:2]
        x0, y0 = (W - rw) // 2, H - rh - 16
        img[y0:y0 + rh, x0:x0 + rw] = cv2.addWeighted(img[y0:y0 + rh, x0:x0 + rw], 0.15, radar, 0.85, 0)
        writer.write(img)
        if rec["f"] == snap_at:
            cv2.imwrite(str(run / "snapshot.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    writer.release()
    print(f"wrote {run / 'annotated.mp4'} and snapshot.jpg")


if __name__ == "__main__":
    main()

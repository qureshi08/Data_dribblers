"""Pitch mapping for small sided football filmed by a camera operator (pans and zooms), where the
11 a side keypoint model does not apply.

1. A few hand marked points on the first analysed frame give a starting homography.
2. Every frame: white line pixels are found (players and broadcast logos masked out), and the court
   template (halfway line, centre circle, goal lines, touchlines) is slid onto them by minimising a
   truncated distance to the nearest line pixel (Nelder-Mead on four image points).
3. Between frames the camera motion comes from ORB features on the background, so the fit follows
   pans and zooms; the line fit then removes drift.
Writes pitch coordinates into tracks.json and records the court size.

    python calibrate.py runs/futsal --size 40x20 --points "17,10:1040,395;20,13:1265,528;20,19:1200,700;0,8.5:483,250;0,11.5:370,272"
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

TRUNC = 12.0          # px, distance beyond which a template point counts as "no line here"


def template(L, W, step=.25):
    pts = []
    for y in np.arange(0, W + 1e-6, step):
        pts += [(L / 2, y), (0, y), (L, y)]
    for x in np.arange(0, L + 1e-6, step):
        pts += [(x, 0), (x, W)]
    for a in np.arange(0, 2 * np.pi, step / 3):
        pts.append((L / 2 + 3 * np.cos(a), W / 2 + 3 * np.sin(a)))
    return np.float32(pts)


def line_dist(frame, boxes, mask_rects):
    """Distance (px) from every pixel to the nearest white line pixel, truncated."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    lines = ((tophat > 28) & (hsv[..., 1] < 70)).astype(np.uint8)
    for x1, y1, x2, y2 in boxes:
        lines[max(0, int(y1) - 4):int(y2) + 4, max(0, int(x1) - 4):int(x2) + 4] = 0
    for x1, y1, x2, y2 in mask_rects:
        lines[y1:y2, x1:x2] = 0
    lines = cv2.morphologyEx(lines, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    d = cv2.distanceTransform(1 - lines, cv2.DIST_L2, 3)
    return np.minimum(d, TRUNC), lines


def project(H, pts):
    p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    return p


def cost(H, tpl, dist):
    h, w = dist.shape
    p = project(H, tpl)
    ok = (p[:, 0] >= 0) & (p[:, 0] < w - 1) & (p[:, 1] >= 0) & (p[:, 1] < h - 1)
    if ok.sum() < 60:
        return 1e9
    d = dist[p[ok, 1].astype(int), p[ok, 0].astype(int)]
    return float(d.mean())


def refine(H, tpl, dist, quad_pitch, iters=220):
    """Nelder-Mead over the image positions of four pitch points."""
    q0 = project(H, quad_pitch).ravel()
    def f(q):
        Hq = cv2.getPerspectiveTransform(quad_pitch, np.float32(q.reshape(4, 2)))
        return cost(Hq, tpl, dist)
    n = len(q0)
    simplex = [q0] + [q0 + np.eye(n)[i] * 8 for i in range(n)]
    vals = [f(s) for s in simplex]
    for _ in range(iters):
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        c = np.mean(simplex[:-1], axis=0)
        xr = c + (c - simplex[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - simplex[-1]); fe = f(xe)
            simplex[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            xc = c + .5 * (simplex[-1] - c); fc = f(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                simplex = [simplex[0]] + [simplex[0] + .5 * (s - simplex[0]) for s in simplex[1:]]
                vals = [vals[0]] + [f(s) for s in simplex[1:]]
    best = simplex[int(np.argmin(vals))]
    return cv2.getPerspectiveTransform(quad_pitch, np.float32(best.reshape(4, 2))), float(min(vals))


def motion(prev, cur, boxes_prev, boxes_cur, mask_rects, orb):
    """Image to image homography (prev -> cur) from background features."""
    def mask(shape, boxes):
        m = np.full(shape[:2], 255, np.uint8)
        for x1, y1, x2, y2 in boxes:
            m[max(0, int(y1) - 6):int(y2) + 6, max(0, int(x1) - 6):int(x2) + 6] = 0
        for x1, y1, x2, y2 in mask_rects:
            m[y1:y2, x1:x2] = 0
        return m
    g1, g2 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    k1, d1 = orb.detectAndCompute(g1, mask(prev.shape, boxes_prev))
    k2, d2 = orb.detectAndCompute(g2, mask(cur.shape, boxes_cur))
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(m) < 20:
        return None
    a = np.float32([k1[x.queryIdx].pt for x in m]); b = np.float32([k2[x.trainIdx].pt for x in m])
    M, inl = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
    return M if M is not None and inl.sum() >= 15 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--size", default="40x20", help="court length x width in metres")
    ap.add_argument("--points", required=True, help='"X,Y:px,py;..." on the first analysed frame')
    ap.add_argument("--mask", default="0,0,140,110;1030,0,1280,110;0,580,160,720",
                    help="x1,y1,x2,y2;... screen areas with broadcast logos")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    L, W = map(float, args.size.split("x"))
    run = Path(args.run)
    data = json.loads((run / "tracks.json").read_text())
    frames = data["frames"]
    rects = [tuple(int(v) for v in r.split(",")) for r in args.mask.split(";") if r]
    tpl = template(L, W)
    quad = np.float32([[L / 2 - 6, W / 2 - 5], [L / 2 + 6, W / 2 - 5], [L / 2 + 6, W / 2 + 5], [L / 2 - 6, W / 2 + 5]])

    pairs = [p.split(":") for p in args.points.split(";")]
    src = np.float32([[float(v) for v in b.split(",")] for _, b in pairs])
    dst = np.float32([[float(v) for v in a.split(",")] for a, _ in pairs])
    H, _ = cv2.findHomography(dst, src, 0)            # pitch -> image

    cap = cv2.VideoCapture(data["video"])
    orb = cv2.ORB_create(1500)
    prev, prev_boxes, costs = None, None, []
    if args.debug:
        (run / "_cal").mkdir(exist_ok=True)
    for i, rec in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, rec["f"])
        ok, img = cap.read()
        if not ok:
            break
        boxes = [p["box"] for p in rec["people"]]
        if prev is not None:
            M = motion(prev, img, prev_boxes, boxes, rects, orb)
            if M is not None:
                H = M @ H
        dist, _ = line_dist(img, boxes, rects)
        H_ref, c = refine(H, tpl, dist, quad)
        if c <= cost(H, tpl, dist):
            H = H_ref
        c = cost(H, tpl, dist)
        costs.append(c)
        Hinv = np.linalg.inv(H)                         # image -> pitch
        rec["h"] = c < 7.0
        for p in rec["people"]:
            x1, y1, x2, y2 = p["box"]
            xy = cv2.perspectiveTransform(np.float32([[[(x1 + x2) / 2, y2]]]), Hinv)[0, 0]
            if rec["h"] and -2 < xy[0] < L + 2 and -2 < xy[1] < W + 2:
                p["xy"] = [round(float(xy[0]), 2), round(float(xy[1]), 2)]
            else:
                p.pop("xy", None)
        if rec["ball"]:
            rec["ball"].pop("xy", None)
            if rec["h"]:
                xy = cv2.perspectiveTransform(np.float32([[rec["ball"]["px"]]]), Hinv)[0, 0]
                if -1 < xy[0] < L + 1 and -1 < xy[1] < W + 1:
                    rec["ball"]["xy"] = [round(float(xy[0]), 2), round(float(xy[1]), 2)]
        if args.debug and i % 24 == 0:
            vis = img.copy()
            for x, y in project(H, tpl).astype(int):
                if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
                    cv2.circle(vis, (x, y), 1, (255, 0, 255), -1)
            cv2.putText(vis, f"cost {c:.1f}", (20, 140), 0, 1, (0, 255, 255), 2)
            cv2.imwrite(str(run / "_cal" / f"{i:04d}.jpg"), vis)
        prev, prev_boxes = img, boxes
    data["pitch"] = [L, W]
    (run / "tracks.json").write_text(json.dumps(data))
    costs = np.array(costs)
    print(f"court {L}x{W} m, median line distance {np.median(costs):.1f} px, mapped {np.mean(costs < 7):.0%} of frames")


if __name__ == "__main__":
    main()

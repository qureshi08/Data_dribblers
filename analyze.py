"""Step 1: match video in, tracking data out.

Detects players, goalkeepers, referees and the ball, tracks each person across frames,
maps every position onto a 105 x 68 m pitch with a homography from detected pitch
keypoints, and splits players into two teams by shirt colour.

    python analyze.py data/match.mp4 --out runs/match
"""
import argparse
import json
import time
from types import SimpleNamespace
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from ultralytics.trackers.bot_sort import BOTSORT

from pitch import fit_homography, on_pitch, to_pitch
from stitch import stitch
from teams import assign

BALL, GK, PLAYER, REF = 0, 1, 2, 3
DATA = Path(__file__).parent / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=0, help="analyse every Nth frame (default: about 6 per second)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--start", type=float, default=0, help="start second")
    ap.add_argument("--seconds", type=float, default=0, help="limit length, 0 = whole video")
    ap.add_argument("--ball-model", action="store_true", help="run the dedicated ball model when the ball is missed")
    ap.add_argument("--colours", default="", help="home and away kit colours from the upload form, e.g. '#ffffff,#a50044'")
    ap.add_argument("--no-pitch", action="store_true", help="skip pitch keypoints (small sided courts the 11 a side model cannot map)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = args.stride or max(1, round(fps / 6))
    first = int(args.start * fps)
    last = min(total, first + int(args.seconds * fps)) if args.seconds else total
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    players_model = YOLO(DATA / "football-player-detection.pt")
    pitch_model = YOLO(DATA / "football-pitch-detection.pt")
    ball_model = YOLO(DATA / "football-ball-detection.pt") if args.ball_model else None
    # BoT-SORT with camera motion compensation: broadcast cameras pan all the time, and without compensating for the
    # pan the predicted boxes slide off the players and tracks break (69 tracks for 22 players, 2026-11-02)
    tracker = BOTSORT(SimpleNamespace(track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.3,
                                      track_buffer=int(fps / stride * 4), match_thresh=0.8, fuse_score=True,
                                      gmc_method="sparseOptFlow", proximity_thresh=0.5, appearance_thresh=0.25,
                                      with_reid=False, model=None))

    frames = []
    classes = defaultdict(Counter)
    H_last, H_age = None, 0
    t0 = time.time()
    print(f"{args.video}: {width}x{height} @ {fps:.1f} fps, frames {first} to {last}, every {stride}")

    for idx in range(first, last):
        if not cap.grab():
            break
        if (idx - first) % stride:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break

        res = players_model.predict(frame, imgsz=args.imgsz, conf=0.1, verbose=False)[0]
        det = sv.Detections.from_ultralytics(res)
        # every ball candidate (both models, low threshold); ball.py later picks the match ball out of them
        cand = det[(det.class_id == BALL) & (det.confidence >= 0.1)]
        if ball_model is not None:
            cand = sv.Detections.merge([cand, sv.Detections.from_ultralytics(ball_model.predict(frame, imgsz=args.imgsz, conf=0.1, verbose=False)[0])])
        balls = cand[cand.confidence >= 0.3]
        boxes = res.boxes.cpu().numpy()
        tracked = tracker.update(boxes[boxes.cls != BALL], frame)
        people = sv.Detections(xyxy=tracked[:, :4].astype(np.float32), confidence=tracked[:, 5],
                               class_id=tracked[:, 6].astype(int), tracker_id=tracked[:, 4].astype(int))             if len(tracked) else sv.Detections.empty()

        kp = None if args.no_pitch else pitch_model.predict(frame, conf=0.3, verbose=False)[0].keypoints
        H = None
        if kp is not None and len(kp.xy) and kp.conf is not None:
            H = fit_homography(kp.xy[0].cpu().numpy(), kp.conf[0].cpu().numpy())
        if H is not None:
            H_last, H_age = H, 0
        else:
            H_age += 1
        H_use = H_last if H_age <= fps / stride * 2 else None   # a stale view is worse than none after 2 s

        rec = {"f": idx, "t": round(idx / fps, 3), "h": H_use is not None, "people": [], "ball": None}
        if H_use is not None:
            rec["H"] = [round(float(v), 8) for v in np.asarray(H_use).ravel()]   # the line-marking mapping: ball.py uses it for touchline calls
        feet = people.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pxy = to_pitch(H_use, feet) if H_use is not None else None
        for i in range(len(people)):
            tid, cls, box = int(people.tracker_id[i]), int(people.class_id[i]), people.xyxy[i]
            classes[tid][cls] += 1
            p = {"id": tid, "box": [round(float(v), 1) for v in box]}
            if pxy is not None:
                p["xy"] = [round(float(pxy[i, 0]), 2), round(float(pxy[i, 1]), 2)]
            rec["people"].append(p)
        rec_balls = [{"px": [round(float((x1 + x2) / 2), 1), round(float((y1 + y2) / 2), 1)], "conf": round(float(c), 3)}
                     for (x1, y1, x2, y2), c in zip(cand.xyxy, cand.confidence)]
        if len(balls):
            b = balls[int(np.argmax(balls.confidence))]
            centre = b.get_anchors_coordinates(sv.Position.CENTER)
            rec["ball"] = {"px": [round(float(centre[0, 0]), 1), round(float(centre[0, 1]), 1)]}
            if H_use is not None:
                bxy = to_pitch(H_use, centre)
                if on_pitch(bxy).all():
                    rec["ball"]["xy"] = [round(float(bxy[0, 0]), 2), round(float(bxy[0, 1]), 2)]
        rec["balls"] = rec_balls   # every candidate; ball.py picks the match ball
        frames.append(rec)

        done = len(frames)
        if done % 20 == 0:
            rate = (time.time() - t0) / done
            left = ((last - idx) / stride) * rate
            print(f"  {done} frames analysed, {rate:.2f} s each, about {left / 60:.1f} min left", flush=True)
    cap.release()

    # majority class per track: player, goalkeeper or referee
    role = {tid: c.most_common(1)[0][0] for tid, c in classes.items()}

    result = {
        "video": str(args.video), "fps": fps, "stride": stride, "width": width, "height": height,
        "frames": frames,
        "tracks": {str(t): {"role": ["ball", "goalkeeper", "player", "referee"][role[t]], "team": -1} for t in role},
        "seconds_processed": round(time.time() - t0, 1),
    }
    team_colour = result["team_colours"] = assign(args.video, frames, result["tracks"], [c for c in args.colours.split(",") if c] or None)
    before = len(result["tracks"])
    result["tracks"] = stitch(frames, result["tracks"])
    print(f"stitched broken tracks: {before} -> {len(result['tracks'])}")
    (out / "tracks.json").write_text(json.dumps(result))
    mapped = sum(r["h"] for r in frames) / max(1, len(frames))
    seen = sum(r["ball"] is not None for r in frames) / max(1, len(frames))
    print(f"done: {len(frames)} frames in {result['seconds_processed']} s, pitch mapped {mapped:.0%}, ball seen {seen:.0%}")
    print(f"teams: {team_colour}, wrote {out / 'tracks.json'}")


if __name__ == "__main__":
    main()

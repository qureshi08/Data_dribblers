"""One command: video in, report out.

    python run.py match.mp4 --names "Arsenal,Chelsea"
    python run.py match.mp4 --seconds 60          # first minute only, for a quick test
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def step(*args):
    print(">", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--names", default="Team A,Team B")
    ap.add_argument("--out", default="")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--halftime", type=float, default=0)
    ap.add_argument("--ball-model", action="store_true")
    ap.add_argument("--no-video", action="store_true", help="skip the annotated video")
    a = ap.parse_args()
    video = Path(a.video).resolve()
    out = Path(a.out) if a.out else HERE / "runs" / video.stem
    extra = ["--ball-model"] if a.ball_model else []
    step("analyze.py", video, "--out", out, "--start", a.start, "--seconds", a.seconds,
         "--stride", a.stride, "--imgsz", a.imgsz, *extra)
    step("ball.py", out)
    step("stats.py", out, "--names", a.names, "--halftime", a.halftime)
    step("advanced.py", out)
    if not a.no_video:
        step("render.py", out)
    step("report.py", out)
    print(f"\nreport: {out / 'report.html'}")


if __name__ == "__main__":
    main()

"""Step 4: stats.json + tracks.json in, one report.html out: a live match centre (video with
the tracking drawn on top, stats and events playing in sync) and the full match report below.

The page plays runs/<name>/video.mp4 (H.264). If that file is missing it is made from the
source video with the ffmpeg bundled in imageio-ffmpeg.

    python report.py runs/match
"""
import argparse
import base64
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).parent


def browser_video(src, dest, width, height):
    """H.264 copy the browser can play. The ffmpeg from imageio-ffmpeg first (small file, clean exit); OpenCV's avc1
    writer only if that is missing (on Windows it writes about 50 Mbps and its Media Foundation teardown can crash the
    process after the file is written, which the app then reads as a failed step).
    An H.264 MP4 or MOV that is already the right size plays in the browser as it is: it is linked (or copied), never
    re-encoded, so a two hour upload takes a second here instead of hours. Times stay absolute either way."""
    import imageio_ffmpeg
    info = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(src)], capture_output=True, text=True).stderr
    if Path(src).suffix.lower() in (".mp4", ".mov", ".m4v") and "Video: h264" in info and "yuv420p" in info and f"{width}x{height}" in info:
        import os
        import shutil
        try:
            os.link(src, dest)   # same drive: no copy, no extra space
        except OSError:
            shutil.copy(src, dest)
        return
    try:
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
                        "-vf", f"scale={width}:{height}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-g", "12",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dest)], check=True)
        return
    except ImportError:
        pass
    import cv2
    cap = cv2.VideoCapture(str(src))
    out = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"avc1"), cap.get(cv2.CAP_PROP_FPS) or 25, (width, height))
    if out.isOpened():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out.write(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
        out.release()
        return
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
                    "-vf", f"scale={width}:{height}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dest)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--event", default="Match analysis", help="competition line in the match bar")
    ap.add_argument("--date", default="", help="match date as shown, e.g. 2 Nov 2026")
    args = ap.parse_args()
    run = Path(args.run)
    stats = json.loads((run / "stats.json").read_text())
    tracks = json.loads((run / "tracks.json").read_text())
    adv = run / "advanced.json"
    stats["adv"] = json.loads(adv.read_text()) if adv.exists() else None

    video = run / "video.mp4"
    if not video.exists():
        try:
            browser_video(tracks["video"], video, tracks["width"], tracks["height"])
        except Exception as e:                       # the static report still works without it
            print(f"no browser video ({e}); the live view will show a still frame")

    frames = []
    for f in tracks["frames"]:
        people = []
        for p in f["people"]:
            row = [p["id"], *p["box"]]
            if "xy" in p:
                row += p["xy"]
            people.append(row)
        frames.append([f["t"], people, f["ball"]["px"] if f["ball"] else None, f["f"]])
    stats["frames"] = frames
    stats["roles"] = {k: [v["team"], v["role"]] for k, v in tracks["tracks"].items()}
    stats["video_size"] = [tracks["width"], tracks["height"]]
    meta_file = run / "match.json"            # remembers --event and --date between rebuilds
    saved = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    match = {"event": args.event if args.event != "Match analysis" else saved.get("event", args.event), "date": args.date or saved.get("date", "")}
    meta_file.write_text(json.dumps(match))
    stats["match"] = match
    stats["video"] = "video.mp4" if video.exists() else ""
    snap = run / "snapshot.jpg"
    stats["snapshot"] = "data:image/jpeg;base64," + base64.b64encode(snap.read_bytes()).decode() if snap.exists() else ""

    html = (HERE / "report_template.html").read_text(encoding="utf-8")
    html = html.replace("/*DATA*/null", json.dumps(stats, separators=(",", ":")).replace("</", "<\\/"))
    (run / "report.html").write_text(html, encoding="utf-8")
    print(f"wrote {run / 'report.html'}")


if __name__ == "__main__":
    main()

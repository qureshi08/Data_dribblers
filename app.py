"""Data Dribblers, the app: upload a match video, it runs the full analysis pipeline (run.py) and opens that match's page.

    .venv/Scripts/python app.py            # then open http://localhost:4820

One job runs at a time (the analysis uses the whole CPU). Every job lives in runs/<id>/ with job.json (settings, stage,
progress, log tail), so the list of matches survives a restart. Video files are served with byte ranges so the match
page can seek. Standard library only.
"""
import json
import mimetypes
import os
import queue
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).parent
RUNS, UPLOADS = HERE / "runs", HERE / "data" / "uploads"
PORT = int(os.environ.get("PORT", 4820))
STAGES = [("upload", "Uploading"), ("analyze", "Finding every player and the ball"), ("ball", "Picking out the match ball"), ("stats", "Counting passes, runs and possession"),
          ("advanced", "Working out space, pressure and lines"), ("report", "Building the match page"), ("done", "Ready")]
MODES = {   # speed against accuracy on a CPU: (detector image size, frames a second analysed, dedicated ball model)
    "fast": (960, 6, False),
    "accurate": (1280, 12, True),   # 12 a second: rings have half as far to guess between samples, the ball twice the chances
}
UPLOADS.mkdir(parents=True, exist_ok=True)
JOBS, LOCK, WORK = {}, threading.Lock(), queue.Queue()


def save(job):
    with LOCK:
        (RUNS / job["id"]).mkdir(parents=True, exist_ok=True)
        (RUNS / job["id"] / "job.json").write_text(json.dumps(job, indent=1), encoding="utf-8")


def load_jobs():
    for d in sorted(RUNS.iterdir()) if RUNS.exists() else []:
        f = d / "job.json"
        if f.exists():
            job = json.loads(f.read_text(encoding="utf-8"))
            if job["stage"] not in ("done", "failed"):
                job.update(stage="failed", error="The app was restarted while this match was being analysed. Upload it again.")
                save(job)
            JOBS[job["id"]] = job
        elif (d / "report.html").exists() and (d / "match.json").exists() and not d.name.startswith("_"):   # titled matches analysed before the app existed
            meta = json.loads((d / "match.json").read_text(encoding="utf-8")) if (d / "match.json").exists() else {}
            JOBS[d.name] = {"id": d.name, "title": meta.get("event", d.name), "teams": "", "stage": "done", "progress": 1,
                            "created": d.stat().st_mtime, "example": True, "report": f"/runs/{d.name}/report.html"}


def slug(name):
    s = re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-")[:32] or "match"
    return f"{s}-{time.strftime('%m%d-%H%M%S')}"


def run_job(job):
    """runs the pipeline step by step, reading each step's output for progress"""
    out, video = RUNS / job["id"], Path(job["video"])
    imgsz, per_sec, ball = MODES[job["mode"]]
    (out / "match.json").write_text(json.dumps({"event": job["title"], "date": job.get("date", "")}), encoding="utf-8")
    fps = probe_fps(video) or 25
    stride = max(1, round(fps / per_sec))
    steps = [("analyze", ["analyze.py", str(video), "--out", str(out), "--seconds", str(job["seconds"]), "--start", str(job["start"]),
                          "--stride", str(stride), "--imgsz", str(imgsz), *(["--ball-model"] if ball else []),
                          *(["--colours", job["colours"]] if job.get("colours") else [])]),
             ("ball", ["ball.py", str(out)]),
             ("stats", ["stats.py", str(out), "--names", job["teams"]]),
             ("advanced", ["advanced.py", str(out)]),
             ("report", ["report.py", str(out), "--event", job["title"], "--date", job.get("date", "")])]
    total = None
    for stage, args in steps:
        job.update(stage=stage, progress=0 if stage == "analyze" else job.get("progress", 0)); save(job)
        p = subprocess.Popen([sys.executable, "-u", *args], cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        tail = []
        for line in p.stdout:
            line = line.rstrip(); tail = (tail + [line])[-12:]
            m = re.search(r"frames (\d+) to (\d+), every (\d+)", line)
            if m: total = max(1, (int(m.group(2)) - int(m.group(1))) // int(m.group(3)))
            m = re.search(r"(\d+) frames analysed, ([\d.]+) s each, about ([\d.]+) min left", line)
            if m and total:
                job.update(progress=min(.99, int(m.group(1)) / total), eta_min=float(m.group(3)), sec_per_frame=float(m.group(2)))
            job["log"] = tail; save(job)
        if p.wait() != 0:
            job.update(stage="failed", error=f"The {stage} step stopped. Last lines: " + " | ".join(tail[-4:])); save(job); return
        if stage == "analyze": job.update(progress=1, eta_min=0)
    job.update(stage="done", progress=1, report=f"/runs/{job['id']}/report.html", finished=time.time()); save(job)


def probe_fps(video):
    try:
        import cv2
        c = cv2.VideoCapture(str(video)); f = c.get(cv2.CAP_PROP_FPS); c.release(); return f or None
    except Exception:  # noqa: BLE001
        return None


def worker():
    while True:
        job = WORK.get()
        try:
            run_job(job)
        except Exception as e:  # noqa: BLE001
            job.update(stage="failed", error=str(e)[:300]); save(job)


class App(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path); path = unquote(u.path)
        if path in ("/", "/index.html"): return self.send_file(HERE / "app.html")
        if path == "/api/jobs":
            jobs = sorted(JOBS.values(), key=lambda j: -j.get("created", 0))
            return self.send_json([{k: j.get(k) for k in ("id", "title", "teams", "stage", "progress", "eta_min", "report", "error", "created", "example", "seconds", "mode", "queue")} for j in jobs])
        m = re.fullmatch(r"/api/jobs/([\w-]+)", path)
        if m:
            j = JOBS.get(m.group(1)); return self.send_json(j) if j else self.send_json({"error": "not found"}, 404)
        if path.startswith("/runs/") or path.startswith("/design/"):
            f = (HERE / path.lstrip("/")).resolve()
            if HERE.resolve() in f.parents and f.is_file(): return self.send_file(f)
        self.send_error(404)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/upload": return self.send_error(404)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        name = unquote(self.headers.get("X-Filename", "match.mp4")); size = int(self.headers.get("Content-Length", 0))
        if not size: return self.send_json({"error": "empty file"}, 400)
        jid = slug(name); dest = UPLOADS / f"{jid}{Path(name).suffix.lower() or '.mp4'}"
        teams = f"{q.get('home') or 'Home'},{q.get('away') or 'Away'}"
        job = {"id": jid, "title": q.get("title") or Path(name).stem, "date": q.get("date", ""), "teams": teams, "file": name,
               "mode": q.get("mode") if q.get("mode") in MODES else "fast",
               "colours": ",".join(c for c in (q.get("home_colour", ""), q.get("away_colour", "")) if re.fullmatch(r"#[0-9a-fA-F]{6}(\+#[0-9a-fA-F]{6})?", c)) if q.get("home_colour") and q.get("away_colour") else "", "seconds": float(q.get("seconds") or 30), "start": float(q.get("start") or 0),
               "stage": "upload", "progress": 0, "created": time.time(), "video": str(dest)}
        JOBS[jid] = job; save(job)
        got = 0
        with open(dest, "wb") as f:
            while got < size:
                chunk = self.rfile.read(min(1 << 20, size - got))
                if not chunk: break
                f.write(chunk); got += len(chunk)
        if got < size:
            job.update(stage="failed", error="The upload was cut off."); save(job); return self.send_json(job, 400)
        job.update(stage="queued", queue=WORK.qsize()); save(job); WORK.put(job)
        self.send_json({"id": jid})

    def send_file(self, f):
        size = f.stat().st_size; ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        rng = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if rng:   # byte ranges, so <video> can seek
            a = int(rng.group(1)) if rng.group(1) else max(0, size - int(rng.group(2)))
            b = int(rng.group(2)) if rng.group(1) and rng.group(2) else size - 1
            b = min(b, size - 1)
            self.send_response(206); self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        else:
            a, b = 0, size - 1; self.send_response(200)
        self.send_header("Content-Type", ctype); self.send_header("Accept-Ranges", "bytes"); self.send_header("Content-Length", str(b - a + 1))
        self.send_header("Cache-Control", "no-cache"); self.end_headers()
        try:
            with open(f, "rb") as fh:
                fh.seek(a); left = b - a + 1
                while left > 0:
                    chunk = fh.read(min(1 << 20, left))
                    if not chunk: break
                    self.wfile.write(chunk); left -= len(chunk)
        except (ConnectionResetError, BrokenPipeError):
            pass


if __name__ == "__main__":
    load_jobs()
    threading.Thread(target=worker, daemon=True).start()
    print(f"Data Dribblers app on http://localhost:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), App).serve_forever()

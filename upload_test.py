"""Uploads a video to the running app (http://localhost:4820) the same way app.html does, streaming the file.

    py upload_test.py <video> --title ... --home BAR --away ATH --home-colour '#a50044+#1d3c87' --away-colour '#d8c6e0' --start 480 --seconds 8 --mode accurate
"""
import argparse
import http.client
import json
import os
from urllib.parse import quote, urlencode

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("--title", default=""); ap.add_argument("--date", default="")
ap.add_argument("--home", default="Home"); ap.add_argument("--away", default="Away")
ap.add_argument("--home-colour", default=""); ap.add_argument("--away-colour", default="")
ap.add_argument("--start", type=float, default=0); ap.add_argument("--seconds", type=float, default=30); ap.add_argument("--mode", default="fast")
a = ap.parse_args()
q = urlencode({"title": a.title, "date": a.date, "home": a.home, "away": a.away, "home_colour": a.home_colour, "away_colour": a.away_colour,
               "start": a.start, "seconds": a.seconds, "mode": a.mode})
size = os.path.getsize(a.video)
c = http.client.HTTPConnection("localhost", 4820, timeout=3600)
c.putrequest("POST", "/api/upload?" + q)
c.putheader("Content-Length", str(size)); c.putheader("X-Filename", quote(os.path.basename(a.video))); c.putheader("Content-Type", "application/octet-stream")
c.endheaders()
with open(a.video, "rb") as f:
    while True:
        chunk = f.read(8 << 20)
        if not chunk:
            break
        c.send(chunk)
print(json.loads(c.getresponse().read()))

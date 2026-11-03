"""Download the three pretrained football models (and one sample clip) from Roboflow's
open source sports repo (github.com/roboflow/sports, examples/soccer/setup.sh).

    python setup_models.py
"""
from pathlib import Path

import gdown

DATA = Path(__file__).parent / "data"
FILES = {
    "football-player-detection.pt": "17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q",
    "football-pitch-detection.pt": "1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf",
    "football-ball-detection.pt": "1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V",
    "sample_0bfacc.mp4": "12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF",   # 30 s Bundesliga broadcast clip, 1080p
}

DATA.mkdir(exist_ok=True)
for name, fid in FILES.items():
    dest = DATA / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"have {name}")
        continue
    gdown.download(id=fid, output=str(dest), quiet=False)

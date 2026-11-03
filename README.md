# Data Dribblers

Match analysis from the video a team already has. Upload a football match recording, and Data Dribblers finds every player and the ball, maps them onto the real pitch, splits the two teams and the officials, and builds a match page: the video with the tracking drawn on top, key moments one tap away, a card for every player, the tactics, and a place to ask the match a question in plain words.

It is built for the teams that have a video but no analyst: academies, school teams, amateur leagues, starting in Pakistan.

## Run it

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # on macOS or Linux: .venv/bin/python
.venv/Scripts/python setup_models.py                         # downloads the three pretrained models into data/
.venv/Scripts/python app.py                                  # then open http://localhost:4820
```

In the app: drop a match video, give the title and the two teams, pick each team's kit colour, drag the green marker on the timeline to kick off, choose how much to analyse (30 s, 1 min, 2 min or the whole video) and Fast or Accurate, then press Analyse match. The page shows each stage with its progress and the minutes left, then opens the match page.

The same pipeline runs without the app:

```
.venv/Scripts/python run.py match.mp4 --names "BAR,ATH" --start 480 --seconds 30 --ball-model
```

## How it works

| Step | File | What it does |
|---|---|---|
| Detect and track | `analyze.py` | YOLOv8 finds players, goalkeepers, referees and ball candidates in each analysed frame. BoT-SORT tracks people across frames with camera motion compensation, so tracks survive the broadcast camera's pans. The pitch keypoint model fits a homography each frame, so every position is in real metres on a 105 x 68 m pitch. |
| Teams | `teams.py` | Groups players by the colour the camera actually shows (median Lab, brightness at half weight), in three groups so neon officials never swallow a team. A second opinion comes from SigLIP image embeddings of each player's shirt and shorts (the method in Roboflow's sports project): where it is sure of a player (75% of his crops in one group) it overrides the colour, which helps when sun and shade change how a kit looks. The kit colours chosen at upload only name the groups. Also catches goalkeepers the detector called referees, and people who stay off the pitch (camera crew, ball boys). |
| Identity | `stitch.py` | Re-joins a player lost for up to 5 s when a new track starts close enough on the pitch and the shirt looks the same. |
| Ball | `ball.py` | Picks the match ball out of every ball candidate: off the pitch is never the ball, spare balls and painted spots that never move are dropped, short tracks are linked within ball speed, and the best physically possible chain is kept. Gaps under 0.6 s are filled. `--slice` also searches enlarged 3 x 2 tiles for a tiny ball; it is off by default, because on Barcelona v Athletic its extra candidates (boots, spots) pulled the path off the real ball. `ball_sheet.py` draws the chosen ball on every frame for checking by eye. |
| Stats | `stats.py` | Possession, passes, shots and xG, distance, speed, sprints, heatmaps, pass network. |
| Tactics | `advanced.py` | Pitch control (Spearman, Shaw simplified), pressing on the ball carrier, xT per pass (Karun Singh's grid), packing, team width, depth and line height. |
| Match page | `report.py`, `report_template.html` | One HTML page: the video with the tracking drawn live, layers (rings, numbers, ball trail, pitch control, pass options, defensive lines, team shape, mini radar), key moments, player cards with a clip behind every number, tactics and Ask. |
| App | `app.py`, `app.html` | Upload, a one job queue, live progress, byte range video serving, the list of analysed matches. Standard library only. |

## Where it stands

Tested on broadcast clips of Bundesliga and LaLiga matches. On a 30 second Bundesliga clip from a fixed wide camera: the pitch mapped in every analysed frame, the two teams and the referee split by kit, the ball found in 77% of frames, possession 65 to 35 and passes counted for both sides. On a LaLiga tactical camera clip (Barcelona v Athletic), with a moving camera, small players and kits that look alike under stadium light, results are weaker. Over 30 seconds of that match (8:08 to 8:38), checked frame by frame with `ball_sheet.py`, the ball was chosen in 62% of frames and was the real ball in nearly all of them. It is lost whenever the director cuts to the wide shot, where the ball is two or three pixels, and a straight line fill misses a lofted pass. The teams split 10 against 10 on the pitch, with the shirt check agreeing with colour on 88% of players. It runs on a laptop processor today, about a minute of computing per second of video in Fast mode; a GPU is the next step for full matches.
Known limits: track numbers are not shirt numbers yet, the ball is lost when it is a few blurred pixels, close ups and replays break tracking, and shots are inferred from ball speed and direction.

## Credits

The three pretrained models (players, pitch keypoints, ball) come from Roboflow's open source sports project, github.com/roboflow/sports. They are downloaded by `setup_models.py` and are not part of this repository, and no match footage is included.

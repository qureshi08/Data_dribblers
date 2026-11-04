"""Fine tunes the ball model on openly licensed footage, for the product.

The pretrained ball model learned from Bundesliga broadcasts and misses a ball of four to eight pixels in a wide shot.
Only data the product may use goes in here: today the Football-Player-Detection set by Augmented Startups (CC BY 4.0,
1,232 wide broadcast frames at 1280 x 720 with the ball boxed, ball about 9 px across), downloaded from Hugging Face
into data/open/keremberke. Competition data (More Than A Hack) never comes into this folder.

  * training images are 640 x 640 crops at full resolution, so the ball keeps the size the app sees when it analyses a
    1280 wide frame, plus crops with no ball in them so boots, spots and line marks are learned as not the ball;
  * frames with no ball box are skipped, since a ball may be there but unlabelled;
  * the set's own test split is never trained on, and `score` runs both models on it at the app's settings.

    python finetune/ball.py prepare
    python finetune/ball.py train [--epochs 3]
    python finetune/ball.py score              # the pretrained model against the tuned one, on the test split

Attribution (CC BY 4.0): Football-Player-Detection Dataset, Augmented Startups, Roboflow Universe, 2022,
universe.roboflow.com/augmented-startups/football-player-detection-kucab
"""
import argparse
import glob
import json
import random
import shutil
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "data" / "open" / "keremberke"
DS = HERE / "data" / "ball_open"
BASE, TUNED = HERE / "data" / "football-ball-detection.pt", HERE / "data" / "ball_open.pt"
CROP = 640


def coco(split):
    """[(image path, width, height, [ball boxes x, y, w, h])] for one split of the source set"""
    j = json.load(open(glob.glob(str(SRC / split / "*.json"))[0]))
    ball = next(c["id"] for c in j["categories"] if c["name"] == "football")
    boxes = {}
    for a in j["annotations"]:
        if a["category_id"] == ball:
            boxes.setdefault(a["image_id"], []).append(a["bbox"])
    return [(SRC / split / i["file_name"], i["width"], i["height"], boxes.get(i["id"], [])) for i in j["images"]]


def yolo_line(b, x0, y0):
    x, y, w, h = b
    return f"0 {(x - x0 + w / 2) / CROP:.6f} {(y - y0 + h / 2) / CROP:.6f} {w / CROP:.6f} {h / CROP:.6f}"


def prepare(a):
    shutil.rmtree(DS, ignore_errors=True); random.seed(7)
    counts = {}
    for split, out in (("train", "train"), ("valid", "val")):
        (DS / "images" / out).mkdir(parents=True); (DS / "labels" / out).mkdir(parents=True)
        pos = neg = 0
        for path, W, H, boxes in coco(split):
            if not boxes:
                continue
            img = cv2.imread(str(path)); stem = path.stem.split("_jpg")[0]
            x, y, w, h = boxes[0]
            x0 = int(min(max(x + w / 2 - random.uniform(60, CROP - 60), 0), W - CROP))
            y0 = int(min(max(y + h / 2 - random.uniform(60, CROP - 60), 0), H - CROP))
            inside = [b for b in boxes if x0 <= b[0] and b[0] + b[2] <= x0 + CROP and y0 <= b[1] and b[1] + b[3] <= y0 + CROP]
            cv2.imwrite(str(DS / "images" / out / f"{stem}.jpg"), img[y0:y0 + CROP, x0:x0 + CROP], [cv2.IMWRITE_JPEG_QUALITY, 95])
            (DS / "labels" / out / f"{stem}.txt").write_text("\n".join(yolo_line(b, x0, y0) for b in inside) + "\n"); pos += 1
            if pos % 3 == 0:   # a crop from the same frame that holds no ball
                for _ in range(10):
                    nx, ny = random.randint(0, W - CROP), random.randint(0, H - CROP)
                    if all(b[0] + b[2] + 30 < nx or b[0] > nx + CROP + 30 or b[1] + b[3] + 30 < ny or b[1] > ny + CROP + 30 for b in boxes):
                        cv2.imwrite(str(DS / "images" / out / f"{stem}_n.jpg"), img[ny:ny + CROP, nx:nx + CROP], [cv2.IMWRITE_JPEG_QUALITY, 95])
                        (DS / "labels" / out / f"{stem}_n.txt").write_text(""); neg += 1
                        break
        counts[out] = (pos, neg)
    (DS / "data.yaml").write_text(f"path: {DS.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
    # the untouched test split, full frames, for `score`
    (DS / "test" / "images").mkdir(parents=True); (DS / "test" / "labels").mkdir(parents=True)
    for path, W, H, boxes in coco("test"):
        shutil.copy(path, DS / "test" / "images" / path.name)
        (DS / "test" / "labels" / f"{path.stem}.txt").write_text("\n".join(
            f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}" for x, y, w, h in boxes) + "\n")
    (DS / "test.yaml").write_text(f"path: {(DS / 'test').as_posix()}\ntrain: images\nval: images\nnames:\n  0: ball\n")
    print(f"ball crops and empty crops: {counts}; test frames: {len(coco('test'))}")


def train(a):
    from ultralytics import YOLO
    YOLO(BASE).train(data=str(DS / "data.yaml"), imgsz=CROP, epochs=a.epochs, batch=8, device="cpu", workers=2, freeze=a.freeze,
                     lr0=.002, warmup_epochs=1, mosaic=.5, scale=.3, fliplr=.5, project=str(HERE / "data" / "ft_runs"), name="ball_open",
                     exist_ok=True, plots=False)
    shutil.copy(HERE / "data" / "ft_runs" / "ball_open" / "weights" / "best.pt", TUNED); print(f"wrote {TUNED}")


def score(a):
    from ultralytics import YOLO
    for name, w in (("pretrained", BASE), ("tuned", TUNED)):
        if not w.exists():
            continue
        m = YOLO(w).val(data=str(DS / "test.yaml"), imgsz=1280, batch=4, device="cpu", conf=.1, plots=False, verbose=False).box
        print(f"{name}: precision {m.mp:.2f}, recall {m.mr:.2f}, mAP50 {m.map50:.2f}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare"); t = sub.add_parser("train"); t.add_argument("--epochs", type=int, default=3); t.add_argument("--freeze", type=int, default=21, help="layers kept fixed; 21 trains only the detection head, about twice as fast on a CPU"); sub.add_parser("score")
    a = ap.parse_args()
    {"prepare": prepare, "train": train, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()

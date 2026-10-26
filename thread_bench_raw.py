# thread_bench_raw.py - compare NCNN inference speed at different thread
# counts, calling ncnn directly (bypasses Ultralytics, which doesn't expose
# net.opt.num_threads in the installed version).
#
# usage (on the Pi): python thread_bench_raw.py best_ncnn_model frames

import sys, time
from pathlib import Path
import cv2
import numpy as np
import ncnn

MODEL_DIR, FRAMES = Path(sys.argv[1]), Path(sys.argv[2])
NAMES = ["unripe", "ripe", "overripe"]
SZ, CONF, IOU = 640, 0.25, 0.7
THREAD_COUNTS = [1, 2, 3, 4]


def letterbox(img):
    h, w = img.shape[:2]
    r = min(SZ / h, SZ / w)
    nh, nw = round(h * r), round(w * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (SZ - nh) // 2, (SZ - nw) // 2
    canvas = np.full((SZ, SZ, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = img
    return canvas


def predict(net, img):
    x = letterbox(img)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    with net.create_extractor() as ex:
        ex.input("in0", ncnn.Mat(np.ascontiguousarray(x)).clone())
        _, out0 = ex.extract("out0")
    out = np.array(out0)
    if out.shape[0] != 7:
        out = out.T
    boxes, scores = out[:4].T, out[4:].T
    conf, cls = scores.max(1), scores.argmax(1)
    keep = conf > CONF
    return int(keep.sum())  # just need detection count for this benchmark


frames = sorted(FRAMES.glob("*.jpg"))
if not frames:
    print(f"No .jpg files found in {FRAMES}")
    sys.exit(1)
imgs = [cv2.imread(str(f)) for f in frames]

print(f"Testing on {len(imgs)} frames\n")

for n_threads in THREAD_COUNTS:
    net = ncnn.Net()
    net.opt.num_threads = n_threads
    net.load_param(str(MODEL_DIR / "model.ncnn.param"))
    net.load_model(str(MODEL_DIR / "model.ncnn.bin"))

    predict(net, imgs[0])  # warm-up

    times = []
    for img in imgs:
        t = time.time()
        predict(net, img)
        times.append(time.time() - t)

    mean_ms = 1000 * sum(times) / len(times)
    fps = len(times) / sum(times)
    print(f"threads={n_threads}: mean {mean_ms:.0f} ms  ({fps:.2f} FPS)")

# tflite_bench.py - benchmark + classify with a TFLite (INT8) model on the Pi
# usage: python tflite_bench.py <model.tflite> frames
#
# This bypasses Ultralytics (which may not support ARM TFLite INT8 cleanly)
# and uses the tflite-runtime interpreter directly, doing the pre/post
# processing by hand — same approach as the earlier NCNN "light_bench.py".

import sys, time
from pathlib import Path
import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

MODEL_PATH, FRAMES = sys.argv[1], Path(sys.argv[2])
NAMES = ["unripe", "ripe", "overripe"]
SZ, CONF, IOU = 640, 0.25, 0.7

interp = Interpreter(model_path=MODEL_PATH)
interp.allocate_tensors()
in_detail = interp.get_input_details()[0]
out_detail = interp.get_output_details()[0]
in_scale, in_zero = in_detail.get("quantization", (0.0, 0))
out_scale, out_zero = out_detail.get("quantization", (0.0, 0))
print(f"Input dtype: {in_detail['dtype']}, quant scale/zero: {in_scale}/{in_zero}")
print(f"Output dtype: {out_detail['dtype']}, quant scale/zero: {out_scale}/{out_zero}\n")


def letterbox(img):
    h, w = img.shape[:2]
    r = min(SZ / h, SZ / w)
    nh, nw = round(h * r), round(w * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (SZ - nh) // 2, (SZ - nw) // 2
    canvas = np.full((SZ, SZ, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = img
    return canvas


def predict(path):
    img = cv2.imread(str(path))
    x = letterbox(img)[:, :, ::-1].astype(np.float32) / 255.0   # HWC, RGB, 0-1

    if in_detail["dtype"] == np.uint8 or in_detail["dtype"] == np.int8:
        x = (x / in_scale + in_zero).astype(in_detail["dtype"])
    x = np.expand_dims(x, axis=0)

    interp.set_tensor(in_detail["index"], x)
    interp.invoke()
    out = interp.get_tensor(out_detail["index"])[0]

    if out_detail["dtype"] in (np.uint8, np.int8):
        out = (out.astype(np.float32) - out_zero) * out_scale

    if out.shape[0] != 7:
        out = out.T
    boxes, scores = out[:4].T, out[4:].T
    conf, cls = scores.max(1), scores.argmax(1)
    keep = conf > CONF
    boxes, conf, cls = boxes[keep], conf[keep], cls[keep]

    results = []
    for c in np.unique(cls):
        m = cls == c
        b = boxes[m]
        xywh = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 2], b[:, 3]], axis=1)
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), conf[m].tolist(), CONF, IOU)
        for i in np.array(idx).flatten():
            results.append((NAMES[c], float(conf[m][i])))
    return sorted(results, key=lambda r: -r[1])


frames = sorted(FRAMES.glob("*.jpg"))
predict(frames[0])   # warm-up
times = []
for f in frames:
    t = time.time()
    dets = predict(f)
    times.append(time.time() - t)
    txt = ", ".join(f"{n} {c:.2f}" for n, c in dets) or "none"
    print(f"{f.name:34s} {1000 * times[-1]:6.0f} ms  {txt}")
print(f"\nmean latency {1000 * sum(times) / len(times):.0f} ms  ({len(times) / sum(times):.1f} FPS)")

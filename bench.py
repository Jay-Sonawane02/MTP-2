import sys, time
from pathlib import Path
from ultralytics import YOLO

model = YOLO(sys.argv[1])
frames = sorted(Path(sys.argv[2]).glob("*.jpg"))
model.predict(str(frames[0]), imgsz=640, verbose=False)   # warm-up
times = []
for f in frames:
    t = time.time()
    r = model.predict(str(f), imgsz=640, conf=0.25, verbose=False)[0]
    times.append(time.time() - t)
    dets = ", ".join(f"{r.names[int(b.cls[0])]} {float(b.conf[0]):.2f}" for b in r.boxes) or "none"
    print(f"{f.name:34s} {1000 * times[-1]:6.0f} ms  {dets}")
print(f"\nmean latency {1000 * sum(times) / len(times):.0f} ms  ({len(times) / sum(times):.1f} FPS)")
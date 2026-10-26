# thread_bench.py - compare NCNN inference speed at different thread counts
# usage (on the Pi): python thread_bench.py best_ncnn_model frames
import sys, time
from pathlib import Path
from ultralytics import YOLO

MODEL_DIR, FRAMES = sys.argv[1], Path(sys.argv[2])
THREAD_COUNTS = [1, 2, 3, 4]

frames = sorted(FRAMES.glob("*.jpg"))
if not frames:
    print(f"No .jpg files found in {FRAMES}")
    sys.exit(1)

print(f"Testing on {len(frames)} frames\n")

for n_threads in THREAD_COUNTS:
    model = YOLO(MODEL_DIR)
    # Ultralytics exposes the underlying ncnn net; set threads before first predict
    try:
        model.predictor = None  # force fresh predictor setup if re-used
    except Exception:
        pass

    model.predict(str(frames[0]), imgsz=640, verbose=False)  # warm-up / build predictor

    # Try to reach into the ncnn net object and set thread count
    set_ok = False
    try:
        net = model.predictor.model.model.net  # path may vary by ultralytics version
        net.opt.num_threads = n_threads
        set_ok = True
    except Exception as e:
        pass

    times = []
    for f in frames:
        t = time.time()
        model.predict(str(f), imgsz=640, conf=0.25, verbose=False)
        times.append(time.time() - t)

    mean_ms = 1000 * sum(times) / len(times)
    fps = len(times) / sum(times)
    tag = "threads set OK" if set_ok else "COULD NOT SET THREADS (see below)"
    print(f"threads={n_threads}: mean {mean_ms:.0f} ms  ({fps:.2f} FPS)  [{tag}]")

print("\nIf 'COULD NOT SET THREADS' appeared, the Ultralytics NCNN wrapper doesn't")
print("expose net.opt directly in this version — tell me and I'll give you a")
print("lower-level script that calls ncnn directly instead of through Ultralytics.")

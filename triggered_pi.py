# triggered_pi.py - run the model only when a tomato is in the chamber
# Pi:     python triggered_pi.py --debug        (manual trigger: touch /tmp/trigger)
# laptop: python triggered_pi.py --model=<best.pt> --cam=1 --device=cpu --debug
#         (manual trigger: create an empty file called "trigger" in your temp folder)
import csv, os, platform, sys, tempfile, time
import cv2
import numpy as np


def arg(name, default):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


MODEL = arg("model", "best_ncnn_model")   # on the laptop: --model=<path to best.pt>
CAM = int(arg("cam", 0))                  # on the laptop: --cam=1 for the chamber webcam
DEVICE = arg("device", "")                # "" = Ultralytics chooses; --device=cpu forces CPU
IMGSZ = 640                 # or (384, 640) after a rectangular re-export
CONF = 0.25
OUT_DIR = "events"
TRIGGER_FILE = os.path.join(tempfile.gettempdir(), "trigger")

POLL_S = 0.2                # how often the cheap presence check runs while idle
CALIB_FRAMES = 15           # empty-chamber frames used at start-up
PRESENT_DELTA = 0.008       # extra saturated-pixel fraction that counts as "something is here"
MOTION_THRESH = 4.0         # frame-to-frame change below this means "settled" (raised from 2.0 — idle noise floor shifted higher after camera/lighting changes)
SETTLE_S = 1.0              # must be present and settled this long before a trigger
VIEWS = 6                   # classifications per event
VIEW_GAP_S = 1.0            # time between views (the roller step goes here)
CLEAR_S = 1.0               # chamber must be empty this long before re-arming
MIN_INTERVAL_S = 3.0        # minimum time between two events

DEBUG = "--debug" in sys.argv


def fresh_frame(cap, drop=2):
    for _ in range(drop):          # skip stale buffered frames without decoding them
        cap.grab()
    ok, f = cap.read()
    return f if ok else None


def small(frame):
    return cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)


def sat_fraction(sm):
    return float((cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)[:, :, 1] > 60).mean())


def blur_gray(sm):
    return cv2.GaussianBlur(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)


def run_event(cap, model, stats):
    """The only place the model runs: VIEWS classifications, VIEW_GAP_S apart, fused into one verdict."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(OUT_DIR, stamp)
    os.makedirs(folder, exist_ok=True)
    scores = {n: 0.0 for n in model.names.values()}
    n_views = 0
    t0 = time.time()
    for v in range(VIEWS):
        if v:
            time.sleep(VIEW_GAP_S)          # settle time; the roller step goes here on the real machine
        frame = fresh_frame(cap)
        if frame is None:
            continue
        cv2.imwrite(os.path.join(folder, f"view_{v}.jpg"), frame)
        t = time.time()
        r = model.predict(frame, imgsz=IMGSZ, conf=CONF, device=DEVICE or None, verbose=False)[0]
        stats["busy"] += time.time() - t
        n_views += 1
        for b in r.boxes:
            scores[model.names[int(b.cls[0])]] += float(b.conf[0])

    total = sum(scores.values())
    verdict = max(scores, key=scores.get) if total > 0 else "none"
    share = scores[verdict] / total if total > 0 else 0.0
    stats["events"] += 1
    took = time.time() - t0
    shown = {k: round(v, 2) for k, v in scores.items()}
    print(f"[{time.strftime('%H:%M:%S')}] EVENT {stats['events']}: {verdict} "
          f"(share {share:.2f}) from {n_views} views | {shown} | {took:.1f} s")

    log = os.path.join(OUT_DIR, "events.csv")
    new = not os.path.exists(log)
    with open(log, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "verdict", "share", "views", "seconds", *scores.keys()])
        w.writerow([stamp, verdict, f"{share:.3f}", n_views, f"{took:.1f}", *[f"{v:.2f}" for v in scores.values()]])


def main():
    from ultralytics import YOLO
    model = YOLO(MODEL)
    linux = platform.system() == "Linux"
    cap = cv2.VideoCapture(CAM, cv2.CAP_V4L2 if linux else cv2.CAP_ANY)
    if linux:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("camera did not open")
        return
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Calibrating: keep the chamber EMPTY...")
    vals = []
    for _ in range(CALIB_FRAMES):
        f = fresh_frame(cap)
        if f is not None:
            vals.append(sat_fraction(small(f)))
        time.sleep(POLL_S)
    baseline = float(np.median(vals))
    print(f"Baseline saturated fraction {baseline:.4f}. Armed. Ctrl+C to stop.")

    stats = {"events": 0, "busy": 0.0}
    state, prev_g = "IDLE", None
    stable_since = clear_since = None
    last_event = 0.0
    start = time.time()
    try:
        while True:
            f = fresh_frame(cap)
            if f is None:
                time.sleep(POLL_S)
                continue
            sm = small(f)
            delta = sat_fraction(sm) - baseline
            present = delta > PRESENT_DELTA
            g = blur_gray(sm)
            motion = float(np.abs(g - prev_g).mean()) if prev_g is not None else 99.0
            prev_g = g
            now = time.time()
            if DEBUG:
                print(f"{state:10s} delta {delta:+.4f} motion {motion:5.2f} present {present}")

            if state == "IDLE":
                manual = os.path.exists(TRIGGER_FILE)
                if manual:
                    os.remove(TRIGGER_FILE)
                stable = present and motion < MOTION_THRESH
                stable_since = (stable_since or now) if stable else None
                ready = stable_since is not None and now - stable_since >= SETTLE_S
                if manual or (ready and now - last_event >= MIN_INTERVAL_S):
                    run_event(cap, model, stats)
                    last_event = time.time()
                    state, prev_g, stable_since, clear_since = "WAIT_CLEAR", None, None, None
            else:   # WAIT_CLEAR: do not fire again until the chamber has been empty for a while
                clear_since = (clear_since or now) if not present else None
                if clear_since is not None and now - clear_since >= CLEAR_S:
                    state, prev_g = "IDLE", None
                    print(f"[{time.strftime('%H:%M:%S')}] chamber clear, armed")
            time.sleep(POLL_S)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        up = time.time() - start
        print(f"\nRan {up:.0f} s, {stats['events']} events, model busy {stats['busy']:.1f} s "
              f"= {100 * stats['busy'] / max(up, 1e-9):.1f}% of the time")


if __name__ == "__main__":
    main()

# demo_capture.py - press ENTER to capture + classify, repeat, Ctrl+C to stop
# usage: python demo_capture.py --model=<path to best.pt> --cam=1 [--device=cpu]
#
# Flow: press ENTER -> takes VIEWS photos (VIEW_GAP_S apart, rotate the tomato
# between them if doing it by hand) -> classifies each -> prints a fused verdict
# -> waits for the next ENTER. Nothing runs between presses.

import os, platform, sys, time
import cv2
import numpy as np


def arg(name, default):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


MODEL = arg("model", r"D:\MTP-2\YOLO_SAM\runs\fullframe_3class_coco_v2\weights\best.pt")
CAM = int(arg("cam", 1))
DEVICE = arg("device", "")       # "" = Ultralytics chooses; --device=cpu forces CPU
IMGSZ = 640
CONF = 0.25
VIEWS = 6                        # photos per capture; rotate the tomato between presses if by hand
VIEW_GAP_S = 1.0                 # pause between views, so you have time to rotate
OUT_DIR = "demo_captures"


def fresh_frame(cap, drop=2):
    for _ in range(drop):        # discard stale buffered frames
        cap.grab()
    ok, f = cap.read()
    return f if ok else None


def main():
    from ultralytics import YOLO
    print(f"Loading model: {MODEL}")
    model = YOLO(MODEL)
    print(f"Classes: {model.names}")

    linux = platform.system() == "Linux"
    cap = cv2.VideoCapture(CAM, cv2.CAP_V4L2 if linux else cv2.CAP_ANY)
    if linux:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        print("camera did not open")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    round_num = 0

    print(f"\nReady. Place a tomato, then press ENTER to capture {VIEWS} views and classify.")
    print("Press Ctrl+C to quit.\n")

    try:
        while True:
            input(">>> Press ENTER to capture...")
            round_num += 1
            stamp = time.strftime("%Y%m%d_%H%M%S")
            folder = os.path.join(OUT_DIR, f"{round_num:02d}_{stamp}")
            os.makedirs(folder, exist_ok=True)

            scores = {n: 0.0 for n in model.names.values()}
            n_views = 0
            t0 = time.time()

            for v in range(VIEWS):
                if v:
                    time.sleep(VIEW_GAP_S)
                frame = fresh_frame(cap)
                if frame is None:
                    print(f"  view {v}: camera read failed, skipping")
                    continue
                cv2.imwrite(os.path.join(folder, f"view_{v}.jpg"), frame)

                t = time.time()
                r = model.predict(frame, imgsz=IMGSZ, conf=CONF, device=DEVICE or None, verbose=False)[0]
                ms = 1000 * (time.time() - t)
                n_views += 1

                view_dets = []
                for b in r.boxes:
                    cls_name = model.names[int(b.cls[0])]
                    conf = float(b.conf[0])
                    scores[cls_name] += conf
                    view_dets.append(f"{cls_name} {conf:.2f}")
                dets_str = ", ".join(view_dets) if view_dets else "no detection"
                print(f"  view {v+1}/{VIEWS}  ({ms:4.0f} ms)  {dets_str}")

            total = sum(scores.values())
            verdict = max(scores, key=scores.get) if total > 0 else "none"
            share = scores[verdict] / total if total > 0 else 0.0
            took = time.time() - t0

            print(f"\n  >>> VERDICT: {verdict.upper()}  (share {share:.2f}, {n_views} views, {took:.1f}s)")
            print(f"      scores: {{{', '.join(f'{k}: {v:.2f}' for k, v in scores.items())}}}")
            print(f"      saved to: {folder}\n")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()

# usage: python live_pi.py usb 30     (or: python live_pi.py cam 30)
import os, sys, time, subprocess
import cv2
from ultralytics import YOLO

SRC = sys.argv[1] if len(sys.argv) > 1 else "usb"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
os.makedirs("captures", exist_ok=True)
model = YOLO("best_ncnn_model")

if SRC == "usb":
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    def grab():
        for _ in range(2):
            cap.read()                      # drop stale buffered frames
        ok, frame = cap.read()
        return frame if ok else None
else:
    def grab():
        subprocess.run(["rpicam-still", "-n", "-t", "300", "--width", "1280", "--height", "720",
                        "-o", "/tmp/cam.jpg"], check=True, capture_output=True)
        return cv2.imread("/tmp/cam.jpg")

for i in range(N):
    frame = grab()
    if frame is None:
        print("no frame from camera")
        break
    cv2.imwrite(f"captures/frame_{i:03d}.jpg", frame)
    t = time.time()
    r = model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
    ms = 1000 * (time.time() - t)
    dets = ", ".join(f"{r.names[int(b.cls[0])]} {float(b.conf[0]):.2f}" for b in r.boxes) or "none"
    print(f"frame {i:03d} | {ms:4.0f} ms | {dets}")
    cv2.imwrite("latest.jpg", r.plot())
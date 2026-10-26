# usage: python stream_pi.py   then open http://tomatopi.local:8000 on your laptop
import threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
from ultralytics import YOLO

MODEL, CAM, PORT, CONF = "best_ncnn_model", 0, 8000, 0.25
COLORS = {"unripe": (0, 255, 0), "ripe": (0, 0, 255), "overripe": (255, 0, 255)}   # BGR

model = YOLO(MODEL)
cap = cv2.VideoCapture(CAM, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_box = {"f": None}
dets = {"list": [], "ms": 0.0}
latest = {"jpg": None}
lock = threading.Lock()

def grabber():                      # keeps the newest camera frame ready
    while True:
        ok, f = cap.read()
        if ok:
            frame_box["f"] = f
        else:
            time.sleep(0.05)

def detector():                     # runs the model on the newest frame (about 2-3 per second)
    while True:
        f = frame_box["f"]
        if f is None:
            time.sleep(0.05)
            continue
        t = time.time()
        r = model.predict(f, imgsz=640, conf=CONF, verbose=False)[0]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            name = r.names[int(b.cls[0])]
            out.append((x1, y1, x2, y2, f"{name} {float(b.conf[0]):.2f}", COLORS.get(name, (255, 255, 255))))
        dets["list"], dets["ms"] = out, 1000 * (time.time() - t)

def streamer():                     # draws the latest boxes on the newest frame, about 10 per second
    while True:
        f = frame_box["f"]
        if f is None:
            time.sleep(0.05)
            continue
        img = f.copy()
        for x1, y1, x2, y2, label, color in dets["list"]:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, label, (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(img, f"inference {dets['ms']:.0f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with lock:
                latest["jpg"] = buf.tobytes()
        time.sleep(0.1)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = b"<html><body style='margin:0;background:#111'><img src='/stream' style='width:100%'></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = None
            try:
                while True:
                    with lock:
                        jpg = latest["jpg"]
                    if jpg is None or jpg is last:
                        time.sleep(0.02)
                        continue
                    last = jpg
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass

for fn in (grabber, detector, streamer):
    threading.Thread(target=fn, daemon=True).start()
print(f"Open http://tomatopi.local:{PORT} on your laptop (same hotspot). Ctrl+C to stop.")
try:
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
except KeyboardInterrupt:
    pass
cap.release()
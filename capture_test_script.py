import cv2
import os
from datetime import datetime

# ─── CONFIG ──────────────────────────────────────────────────────────
CAPTURE_CLASS = "unripe"          # change to "unripe" for that session
CAPTURE_ROOT  = r"D:\MTP-2\chamber_capture_TEST"
CAMERA_INDEX  = 1
POSITIONS_PER_FRUIT = 6

def capture_session():
    save_dir = os.path.join(CAPTURE_ROOT, CAPTURE_CLASS)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Camera failed to open — try changing CAMERA_INDEX to 0, 1, or 2")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    tomato_id = 0
    pos = 0
    total = 0

    print(f"=== TEST SET Capture: {CAPTURE_CLASS} ===")
    print("SPACE = capture | N = next tomato | Q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        cv2.putText(display, f"[TEST SET] Tomato {tomato_id} | Pos {pos}/{POSITIONS_PER_FRUIT}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Capture - TEST SET", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            ts = datetime.now().strftime("%H%M%S")
            path = os.path.join(save_dir, f"{CAPTURE_CLASS}_{tomato_id:03d}_p{pos}_{ts}.jpg")
            cv2.imwrite(path, frame)
            total += 1
            pos += 1
            print(f"  saved — total: {total}")
            if pos >= POSITIONS_PER_FRUIT:
                print(f"Tomato {tomato_id} done. Load next, press N.")

        elif key == ord('n'):
            tomato_id += 1
            pos = 0
            print(f"\n-- Tomato {tomato_id} --")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone — {tomato_id} tomatoes, {total} images saved to {save_dir}")

if __name__ == "__main__":
    capture_session()
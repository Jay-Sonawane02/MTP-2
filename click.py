import cv2

IMAGE_PATH = r"D:\MTP-2\chamber_capture\unripe\unripe_000_p0_210218.jpg"

points = []
img = cv2.imread(IMAGE_PATH)
img_display = img.copy()

def click_event(event, x, y, flags, param):
    global img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")
        cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
        if len(points) % 2 == 0:
            cv2.rectangle(img_display, points[-2], points[-1], (0, 255, 0), 2)
        cv2.imshow("Click TL then BR for DIRECT box, then TL/BR for MIRROR box", img_display)

cv2.imshow("Click TL then BR for DIRECT box, then TL/BR for MIRROR box", img_display)
cv2.setMouseCallback("Click TL then BR for DIRECT box, then TL/BR for MIRROR box", click_event)

print("Click 4 points total:")
print("1. Top-left of a generous box around the DIRECT tomato region")
print("2. Bottom-right of that box")
print("3. Top-left of a generous box around the MIRROR tomato region")
print("4. Bottom-right of that box")
print("Press Q when done.")

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

if len(points) >= 4:
    direct_box = [points[0][0], points[0][1], points[1][0], points[1][1]]
    mirror_box = [points[2][0], points[2][1], points[3][0], points[3][1]]
    print(f"\nDIRECT_BOX = {direct_box}")
    print(f"MIRROR_BOX = {mirror_box}")
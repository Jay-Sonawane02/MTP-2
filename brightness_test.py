import cv2
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
for _ in range(10): cap.read()
ok, f = cap.read(); cap.release()
cv2.imwrite(r"D:\MTP-2\empty_check.jpg", f)
print(f.shape, cv2.cvtColor(f[820:1000, 1160:1450], cv2.COLOR_BGR2HSV)[:, :, 2].mean().round(1))
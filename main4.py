import cv2 as cv
import numpy as np

cap = cv.VideoCapture("bang_chuyen.mp4")
count = 0
vat_the = []
line_x = 100

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    gray = cv.medianBlur(gray, 5)

    circles = cv.HoughCircles(
        gray,
        cv.HOUGH_GRADIENT,
        dp = 1,
        minDist = 10,
        param1= 50,
        param2= 50,
        minRadius=10,
        maxRadius=50
    )

    if circles is not None:
        circles = np.uint16( np.round(circles))
        for x, y, r in circles[0]:
            cv.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv.circle(frame, (x, y), 2, (0, 0, 255), 3)
            


    cv.imshow("f", frame)
    if cv.waitKey(100) == ord('q'):
        break
cv.destroyAllWindows()
import cv2
import os
import sys

cap = cv2.VideoCapture(0)

start = False
count = 0

while True:
    ret,frame = cap.read()
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 255, 255), 2)

    if start:
        roi = frame[100:300, 100:300]
        save_path = "C://Users//a//Desktop//VScode//Rock Paper Scissor//My Game//Images//Scissor//"+str(count)+".jpg"
        cv2.imwrite(save_path, roi)
        count = count+1
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Collecting {}".format(count),
            (5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting images", frame)

    k = cv2.waitKey(100)
    if k == ord('a'):
        start = not start

    if k == ord('q'):
        break

print("\n{} image(s) saved to {}".format(count, IMG_CLASS_PATH))
cap.release()
cv2.destroyAllWindows()

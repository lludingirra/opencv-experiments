import cv2
import math
import cvzone
import random
import time
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  
cap.set(4, 720)   

detector = HandDetector(detectionCon=0.8, maxHands=2)

x_values = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y_values = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x_values, y_values, 2)

cx, cy = 250, 250
color = (255, 0, 255)
counter = 0
score = 0
timeStart = time.time()
totalTime = 30

while True:
    success, img = cap.read()
    if not success:
        print("Unable to capture camera image!")
        break

    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img, flipType=False, draw=False)
    
    if time.time() - timeStart < totalTime:

        if hands:
            lmList = hands[0]['lmList']

            if len(lmList) > 17:
                x1, y1 = lmList[5][:2]
                x2, y2 = lmList[17][:2]

                distance = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
                A, B, C = coff
                distanceCM = A * distance ** 2 + B * distance + C

                x, y, w, h = hands[0]['bbox']

                if distanceCM < 40:
                    if x < cx < x + w and y < cy < y + h:
                        counter = 1

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
                cvzone.putTextRect(img, f'{int(distanceCM)} cm', (x + 5, y - 10))
            else:
                print("Hand coordinates were not detected correctly!")

        if counter:
            counter += 1
            color = (0, 255, 0)

            if counter == 3:
                cx = random.randint(100, 1100)
                cy = random.randint(100, 600)
                color = (255, 0, 255)
                score += 1
                counter = 0

        cv2.circle(img, (cx, cy), 30, color, cv2.FILLED)
        cv2.circle(img, (cx, cy), 10, (255, 255, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 20, (255, 255, 255), 2)
        cv2.circle(img, (cx, cy), 30, (50, 50, 50), 2)

        cvzone.putTextRect(img, f'Time: {int(totalTime - (time.time() - timeStart))}',
                           (1000, 75), scale=3, offset=20)
        cvzone.putTextRect(img, f'Score: {str(score).zfill(2)}', (60, 75),
                           scale=3, offset=20)

    else:
        cvzone.putTextRect(img, 'Game Over', (400, 400), scale=5, offset=30, thickness=7)
        cvzone.putTextRect(img, f'Your Score: {score}', (450, 500), scale=3, offset=20)
        cvzone.putTextRect(img, 'Press R to restart', (460, 575), scale=2, offset=10)

    cv2.imshow("Hand Tracking", img)

    key = cv2.waitKey(1) & 0xFF  

    if key == ord('q'):
        break
    
    if key == ord('r'):  
        timeStart = time.time()
        score = 0

cap.release()
cv2.destroyAllWindows()

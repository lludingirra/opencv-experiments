import cv2
import math
import numpy as np
import cvzone
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  
cap.set(4, 720)   

detector = HandDetector(detectionCon=0.8, maxHands=1)

x_values = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y_values = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x_values, y_values, 2)

frame_counter = 0

while True:
    success, img = cap.read()
    if not success:
        print("Kamera görüntüsü alınamıyor!")
        break

    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img, flipType=False, draw=False)

    if hands:
        lmList = hands[0]['lmList']

        if len(lmList) > 17: 
            x1, y1 = lmList[5][:2]
            x2, y2 = lmList[17][:2]

            distance = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            A, B, C = coff
            distanceCM = A * distance ** 2 + B * distance + C

            x, y, w, h = hands[0]['bbox']
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cvzone.putTextRect(img, f'{int(distanceCM)} cm', (x + 5, y - 10), scale=2, thickness=2)
        else:
            print("El koordinatları doğru algılanmadı!")

    else:
        print("El algılanamadı.")

    # Görüntüyü göster
    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

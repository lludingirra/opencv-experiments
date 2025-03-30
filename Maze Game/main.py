import cv2
import math
import numpy as np
import cvzone
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720 )

color = (210, 204, 5)
detector = HandDetector(detectionCon=0.8)  
cx, cy, w, h = 100, 100, 200, 200  

class DragRect() :
    
    def __init__(self, posCenter, size=[200,200]) :
        
        self.posCenter = posCenter
        self.size = size
        
    def update(self, cursor) :
        
        cx, cy = self.posCenter
        w, h = self.size
        
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
            cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor[:2] 

rectList = []
for x in range(5) :
    
    rectList.append(DragRect([x*250+150,150]))

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1) 

    if not success:
        print("Unable to capture camera image!")
        break
    hands, img = detector.findHands(img)

    if hands:
        lmList = hands[0]['lmList']  
        x1, y1 = lmList[8][0], lmList[8][1] 
        x2, y2 = lmList[12][0], lmList[12][1] 

        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if distance < 30:  
            if lmList and len(lmList) > 8:
                cursor = lmList[8]
                for rect in rectList :
                    rect.update(cursor)
        
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList :
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2), 
                      (cx + w // 2, cy + h // 2), color, cv2.FILLED)
        
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)
    
    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1-alpha, 0)[mask]

    cv2.imshow("IMG", out)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()

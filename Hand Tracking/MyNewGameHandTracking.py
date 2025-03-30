import cv2
import mediapipe as mp
import time
import HandTrackingModulee as htm

pTime = 0
cap = cv2.VideoCapture(0)
detector = htm.HandDetector()

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Unable to capture camera image!")
        break
    
    img = cv2.flip(img, 1)  
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0:
        print("Thumb:", lmList[4]) 
    
    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime - pTime > 0 else 0
    pTime = cTime
    
    cv2.putText(img, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    
    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
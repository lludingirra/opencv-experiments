import cv2
import time 
import numpy as np
import HandTrackingModule as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 488
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(3, hCam)
pTime = 0

detector = htm.HandDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volPer = 0
volBar = 400

while True :
    
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0 :
        #print(lmList[2],lmList[8])
        
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx,cy = (x1+x2) // 2, (y1+y2) // 2
        
        cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 15, (255,0,255), cv2.FILLED)
        cv2.line(img, (x1,y1), (x2,y2), (255,0,0), 3)
        cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)
        
        lenght = math.hypot(x2-x1, y2-y1)
        
        vol = np.interp(lenght, [50,300], [minVol, maxVol])
        volBar = np.interp(lenght, [50,300], [400, 150])
        volPer = np.interp(lenght, [50,300], [0, 100])


        volume.SetMasterVolumeLevel(vol, None)
         
        
        if lenght < 50 :
            
            cv2.circle(img, (cx,cy), 15, (0,255,0), cv2.FILLED)
            
        cv2.rectangle(img, (50,150), (85,400), (0,255,0), 3)
        cv2.rectangle(img, (50,int(volBar)), (85,400), (0,255,0), cv2.FILLED)
        cv2.putText(img,f'{int(volPer)}%', (40,450), cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,255,0),2)

    
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(img,f'FPS: {int(fps)}', (40,50), cv2.FONT_HERSHEY_COMPLEX,
                1,(255,0,255),2)
    
    cv2.imshow("IMG", img)
    key = cv2.waitKey(1)
    
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
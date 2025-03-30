import cv2
import time
import mediapipe as mp

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode, 
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon, 
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []  # Initialize lmList
        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]

                for idd, lm in enumerate(myHand.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([idd, cx, cy])

                    if draw:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return self.lmList
    
    def fingersUp(self):
        fingers = []

        if len(self.lmList) == 0 or len(self.tipIds) < 5:
            return [0, 0, 0, 0, 0]  # If lmList is empty, return all fingers down (0)
        
        # Check if the thumb is up or down (thumb has a different movement)
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:  
            fingers.append(1)  # Thumb is up
        else:
            fingers.append(0)  # Thumb is down

        # Check other fingers (1 to 4)
        for i in range(1, 5):
            if self.lmList[self.tipIds[i]][2] < self.lmList[self.tipIds[i] - 2][2]:  
                fingers.append(1)  # Finger is up
            else:
                fingers.append(0)  # Finger is down

        return fingers

def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Unable to capture camera image!")
            break
        
        img = cv2.flip(img, 1)  # Flip the image horizontally for a mirror effect
        img = detector.findHands(img)  # Detect hands
        lmList = detector.findPosition(img)  # Find the position of landmarks
        
        if len(lmList) != 0:
            # Example: Accessing the thumb landmark (4)
            print("Thumb:", lmList[4])  # Debug line to check thumb position
            
            # Get fingers up status
            fingers = detector.fingersUp()
            print("Fingers up:", fingers)  # Debug line to check finger status
        
        # Calculate FPS (frames per second)
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime - pTime > 0 else 0
        pTime = cTime
        
        # Display FPS on the image
        cv2.putText(img, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        
        # Show the image
        cv2.imshow("Hand Tracking", img)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

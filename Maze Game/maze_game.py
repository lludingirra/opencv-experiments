import cv2
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

color_rect = (210, 204, 5)  
color_circle = (0, 0, 255)  
color_finish = (0, 255, 0) 
detector = HandDetector(detectionCon=0.8)

class DragRect:
    def __init__(self, posCenter, size=[100, 100]):
        self.posCenter = posCenter
        self.size = size

    def check_collision(self, circle_center, radius):
        cx, cy = self.posCenter
        w, h = self.size
        closest_x = max(cx - w // 2, min(circle_center[0], cx + w // 2))
        closest_y = max(cy - h // 2, min(circle_center[1], cy + h // 2))
        distance = math.sqrt((closest_x - circle_center[0]) ** 2 + (closest_y - circle_center[1]) ** 2)
        return distance < radius

class DragCircle:
    def __init__(self, posCenter, radius=30):
        self.start_pos = posCenter
        self.posCenter = posCenter
        self.radius = radius
        self.grabbed = False  
    def update(self, cursor):
        if self.grabbed:
            self.posCenter = cursor[:2]

def reset_game():
    global game_over, game_won
    game_over = False
    game_won = False
    circle.posCenter = circle.start_pos  

rect_positions = [
    (200, 200), (400, 200), (600, 200), (800, 200), (1000, 200),
    (200, 400), (400, 400), (800, 400), (1000, 400),
    (200, 600), (600, 600), (1000, 600)
]

rectList = [DragRect(pos) for pos in rect_positions]

circle = DragCircle([640, 360])

finish_pos = (1100, 100)
finish_radius = 40

game_over = False
game_won = False

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if not success:
        print("Unable to capture camera image!")
        break

    hands, img = detector.findHands(img)

    if hands and not game_over and not game_won: 
        lmList = hands[0]['lmList']
        if len(lmList) >= 13: 
            x1, y1 = lmList[8][:2]  
            x2, y2 = lmList[12][:2]  

        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if distance < 30:  
            cursor = lmList[8]

            if math.sqrt((cursor[0] - circle.posCenter[0]) ** 2 +
                         (cursor[1] - circle.posCenter[1]) ** 2) < circle.radius:
                circle.grabbed = True

        else:
            circle.grabbed = False  

        if circle.grabbed:
            circle.update(cursor)

    imgNew = np.zeros_like(img, np.uint8)

    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                      (cx + w // 2, cy + h // 2), color_rect, cv2.FILLED)

    cv2.circle(img, circle.posCenter, circle.radius, color_circle, cv2.FILLED)

    cv2.circle(img, finish_pos, finish_radius, color_finish, cv2.FILLED)

    for rect in rectList:
        if rect.check_collision(circle.posCenter, circle.radius):
            game_over = True

    if math.sqrt((circle.posCenter[0] - finish_pos[0]) ** 2 +
                 (circle.posCenter[1] - finish_pos[1]) ** 2) < (circle.radius + finish_radius):
        game_won = True

    if game_over:
        cv2.putText(imgNew, "GAME OVER! Press 'R' to Restart", (350, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)
    elif game_won:
        cv2.putText(imgNew, "YOU WIN! Press 'R' to Restart", (400, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)

    out = img.copy()
    alpha = 0.1  
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("IMG", out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  
        break
    elif key == ord('r'):  
        reset_game()

cap.release()
cv2.destroyAllWindows()

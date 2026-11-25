import cv2
import pyautogui
import mediapipe as mp

cap=cv2.VideoCapture(0)
mp_hands=mp.solutions.hands
hands=mp_hands.Hands()
draw=mp.solutions.drawing_utils

while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=hands.process(rgb)
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            draw.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)
            lm=hand.landmark
            fingers=[
                lm[8].y<lm[6].y,
                lm[12].y<lm[10].y,
                lm[16].y<lm[14].y,
                lm[20].y<lm[18].y
            ]
            
            if all (fingers):
                pyautogui.scroll(5)
                status="PALM OPEN - SCROLL UP"
            elif not any(fingers):
                pyautogui.scroll(-5)
                status="FIST - SCROLL DOWN"
            else:
                status="HAND IDLE"

            cv2.putText(frame,status,(30,50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    cv2.imshow("Hand Scroller",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np

mp_hands=mp.solutions.hands
hands=mp_hands.Hands()
mp_draw=mp.solutions.drawing_utils
cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    if not success:
            break
    img=cv2.flip(img,1)
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(rgb)
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            ring=hand.landmark[16]
            index=hand.landmark[8]
            pinky=hand.landmark[20]
            h,w,_=img.shape
            y1=int(ring.y*h)
            y2=int(index.y*h)
            y3=int(pinky.y*h)
            if y1>y2 and y1>y3:
                color="red"
            elif y2>y1 and y2>y3:
                color="green"
            elif y3>y1 and y3>y2:
                color="blue"
            mp_draw.draw_landmarks(img,hand,mp_hands.HAND_CONNECTIONS)
    if color=="red":
        red=np.zeros_like(img)
        red[:,:,2]=255
        output=cv2.addWeighted(img,0.7,red,0.3,0)
    elif color=="green":
        green=np.zeros_like(img)
        green[:,:,1]=255
        output=cv2.addWeighted(img,0.7,green,0.3,0)
    elif color=="blue":
        blue=np.zeros_like(img)
        blue[:,:,0]=255
        output=cv2.addWeighted(img,0.7,blue,0.3,0)
    cv2.imshow("Hand Tint Controler",img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
        
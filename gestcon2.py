import cv2
import mediapipe as mp
import screen_brightness_control as sbc
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
    result=hands.process(rgb)
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img,hand,mp_hands.HAND_CONNECTIONS)
            thumb=hand.landmark[4]
            index=hand.landmark[8]
            h,w,_=img.shape
            x1,y1=int(thumb.x*w),int(thumb.y*h)
            x2,y2=int(index.x*w),int(index.y*h)
            distance=np.hypot(x2-x1,y2-y1)
            brightness=int(np.interp(distance,[20,100],[0,100]))
            brightness=max(0,min(100,brightness))
            sbc.set_brightness(brightness)
            if brightness==0:
                intensity="None"
            elif brightness>0 and brightness<=33:
                intensity="Low"
            elif brightness>33 and brightness<=66:
                intensity="Medium"
            elif brightness>66 and brightness<=100:
                intensity="High"
            cv2.putText(img,f"Brightness: {intensity}",(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.line(img,(x1,y1),(x2,y2),(255,0,0))
    cv2.imshow("Gesture Brightness Control",img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
                break
cap.release()
cv2.destroyAllWindows()

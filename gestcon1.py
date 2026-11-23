import cv2
import mediapipe as mp
import math
from pycaw.pycaw import AudioUtilities


devices=AudioUtilities.GetSpeakers()
volume=devices.EndpointVolume


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
            thumb=hand.landmark[4]
            index=hand.landmark[8]
            h,w,_=img.shape
            x1,y1=int(thumb.x*w),int(thumb.y*h)
            x2,y2=int(index.x*w),int(index.y*h)
            distance=math.hypot(x2-x1,y2-y1)
            vol=min(max(distance,30),200)
            vol=(vol-30)/(200-30)
            volume.SetMasterVolumeLevelScalar(vol,None)
            cv2.circle(img,(x1,y1),10,(255,0,0),-1)
            cv2.circle(img,(x2,y2),10,(255,0,0),-1)
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3)
            cv2.putText(img,f"Volume : {int(vol*100)}%",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            mp_draw.draw_landmarks(img,hand,mp_hands.HAND_CONNECTIONS)
    cv2.imshow("Hand Volume Control",img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
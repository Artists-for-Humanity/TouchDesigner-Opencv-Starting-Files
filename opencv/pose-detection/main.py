import cv2
import mediapipe as mp
import time
from pythonosc import udp_client

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(2)
# cap = cv2.VideoCapture('dancing.mp4')
pTime = 0

client = udp_client.SimpleUDPClient("127.0.0.1", 5005)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)

    if results.pose_landmarks:
        # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        # print(results.pose_landmarks.landmark[mpPose.PoseLandmark.NOSE].x)

        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)

            print(f"{lm.y} {h} {cy}")
            # client.send_message(f"/landmark-{id}-x", lm.x)
            # client.send_message(f"/landmark-{id}-y", lm.y)

            # Check this out for how to break each specific body part out: https://github.com/Gidrian/Multiple-person-detection-mediapipe/blob/main/Multiple_Human_Pose_Detection_Mediapipe/Human_Detection.py
            client.send_message(f"/landmark-{id}-x", cx)
            client.send_message(f"/landmark-{id}-y", cy)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # cv2.putText(img, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

# Define constants
VOL_MIN, VOL_MAX = 0, 100
FINGER_TRACKING_RANGE = (30, 350)
VOLBAR_RANGE = (400, 150)

def main():
    # Initialize camera and MediaPipe hands
    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    # Initialize audio devices
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume.iid, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    try:
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to capture frame")
                break

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            lmList = []
            if results.multi_hand_landmarks:
                for handlandmark in results.multi_hand_landmarks:
                    for id, lm in enumerate(handlandmark.landmark):
                        h, w, _ = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])
                    mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

            if lmList:
                x1, y1 = lmList[4][1], lmList[4][2]  # thumb
                x2, y2 = lmList[8][1], lmList[8][2]  # index finger

                cv2.circle(img, (x1, y1), 13, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 13, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

                length = hypot(x2 - x1, y2 - y1)
                vol = np.interp(length, FINGER_TRACKING_RANGE, (VOL_MIN, VOL_MAX))
                volbar = np.interp(length, FINGER_TRACKING_RANGE, VOLBAR_RANGE)
                volper = np.interp(length, FINGER_TRACKING_RANGE, (0, 100))

                print(vol, int(length))
                volume.SetMasterVolumeLevel(vol, None)

                cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 4)
                cv2.rectangle(img, (50, int(volbar)), (85, 400), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, f"{int(volper)}%", (10, 40), cv2.FONT_ITALIC, 1, (0, 255, 98), 3)
                cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            print()
    except KeyboardInterrupt:
        print("Keyboard Interrupt. Releasing resources.")
    finally:
        
        cap.release()
        cv2.destroyAllWindows()
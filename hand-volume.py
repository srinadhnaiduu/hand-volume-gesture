import cv2
import mediapipe as mp
import math
import ctypes
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Constants
MIN_VOLUME = 0
MAX_VOLUME = 100
VOLUME_BAR_RANGE = 400
FINGER_TRACKING_RANGE = 300

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

# Initialize audio devices
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def main():
    current_volume = 50  # Start at a mid-level volume
    volume.SetMasterVolumeLevelScalar(current_volume / 100, None)

    while True:
        # Capture frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Convert frame to RGB and process using MediaPipe hands
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw hand landmarks on original image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Calculate distance between thumb and index finger tips
                thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1]
                thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0]
                index_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]
                index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0]
                distance = calculate_distance(thumb_tip_x, thumb_tip_y, index_tip_x, index_tip_y)

                # Adjust volume based on hand movement
                if distance < 50:  # Threshold for increasing volume
                    current_volume = min(current_volume + 1, MAX_VOLUME)
                elif distance > 100:  # Threshold for decreasing volume
                    current_volume = max(current_volume - 1, MIN_VOLUME)

                # Set master volume level
                volume.SetMasterVolumeLevelScalar(current_volume / 100, None)

                # Display volume level as percentage
                cv2.putText(frame, f"Volume: {current_volume}%", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Update volume bar
                volume_bar_x = int((current_volume / MAX_VOLUME) * VOLUME_BAR_RANGE)
                cv2.rectangle(frame, (10, 30), (10 + volume_bar_x, 40), (0, 255, 0), cv2.FILLED)

        # Display frame
        cv2.imshow("Hand Volume Control", frame)

        # Wait for key press (space bar) to exit
        if cv2.waitKey(1) & 0xFF == 32:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

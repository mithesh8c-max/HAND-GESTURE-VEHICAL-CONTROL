import cv2
import mediapipe as mp
import serial
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# BLUETOOTH
# =========================

BLUETOOTH_PORT = "COM5"
BAUD_RATE = 9600

try:
    Bluetooth = serial.Serial(
        BLUETOOTH_PORT,
        BAUD_RATE,
        timeout=1
    )

    time.sleep(2)

    print("Bluetooth connected on", BLUETOOTH_PORT)

except Exception as e:
    print("Bluetooth connection failed:")
    print(e)
    exit()

# =========================
# MEDIAPIPE
# =========================

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)

# =========================
# CAMERA
# =========================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera error")
    Bluetooth.close()
    exit()

timestamp = 0

last_command = ""

# =========================
# SEND BLUETOOTH COMMAND
# =========================

def send_command(command):

    global last_command

    if command != last_command:

        Bluetooth.write(command.encode())

        print("Sent:", command)

        last_command = command


# =========================
# MAIN LOOP
# =========================

while True:

    success, frame = camera.read()

    if not success:
        print("Camera error")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp += 1

    result = detector.detect_for_video(
        mp_image,
        timestamp
    )

    # Default = STOP
    gesture = "STOP"
    command = "S"

    # =========================
    # HAND DETECTED
    # =========================

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        wrist = hand[0]
        index = hand[8]

        dx = index.x - wrist.x
        dy = index.y - wrist.y

        # =========================
        # GESTURE DETECTION
        # =========================

        if abs(dx) > abs(dy):

            if dx < -0.20:
                gesture = "LEFT"
                command = "L"

            elif dx > 0.20:
                gesture = "RIGHT"
                command = "R"

        else:

            if dy < -0.20:
                gesture = "FORWARD"
                command = "F"

            elif dy > 0.20:
                gesture = "BACKWARD"
                command = "B"

        # =========================
        # DRAW LANDMARKS
        # =========================

        h, w, _ = frame.shape

        for point in hand:

            x = int(point.x * w)
            y = int(point.y * h)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

    # =========================
    # SEND COMMAND
    # =========================

    send_command(command)

    # =========================
    # DISPLAY
    # =========================

    cv2.putText(
        frame,
        "GESTURE: " + gesture,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "COMMAND: " + command,
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Hand Gesture Robot",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# CLEAN UP
# =========================

send_command("S")

Bluetooth.close()
camera.release()
detector.close()
cv2.destroyAllWindows()

print("Program stopped.")
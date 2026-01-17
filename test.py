import cv2
import mediapipe as mp
import numpy as np
import math
import platform
import subprocess
import time
import threading
import pyautogui  # ← added for cursor & click

# ================= OS DETECTION =================
OS = platform.system()

# ================= WINDOWS VOLUME =================
if OS == "Windows":
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    devices = AudioUtilities.GetSpeakers()
    interface = devices._comobj.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    min_vol, max_vol = volume.GetVolumeRange()[:2]

# ================= MAC VOLUME =================
def set_mac_volume(vol):
    subprocess.call(
        ["osascript", "-e", f"set volume output volume {int(vol)}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def set_volume_async(vol):
    threading.Thread(
        target=set_mac_volume if OS == "Darwin" else set_windows_volume,
        args=(vol,),
        daemon=True
    ).start()

def set_windows_volume(vol):
    sys_vol = np.interp(vol, [0, 100], [min_vol, max_vol])
    volume.SetMasterVolumeLevel(sys_vol, None)

# ================= UTILS =================
def distance(p1, p2):
    return math.hypot(p2[1] - p1[1], p2[2] - p1[2])


def main():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # ===== ORIGINAL VOLUME VARIABLES =====
    prev_length = 0
    alpha = 0.25
    last_sent_volume = -1
    DEAD_ZONE = 3
    MIN_DIST = 40
    MAX_DIST = 240
    # ===== VOLUME BAR UI =====
    BAR_X = 1150
    BAR_Y_TOP = 150
    BAR_Y_BOTTOM = 550
    BAR_WIDTH = 30


    # ===== MOUSE STATE =====
    screen_w, screen_h = pyautogui.size()
    last_click_time = 0
    CLICK_DELAY = 0.8

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                lm = []

                for i, p in enumerate(hand_landmarks.landmark):
                    lm.append([i, int(p.x * w), int(p.y * h)])

                # ===== FINGER STATES =====
                thumb_up = lm[4][1] < lm[3][1]
                index_up = lm[8][2] < lm[6][2]
                middle_down = lm[12][2] > lm[10][2]
                ring_down = lm[16][2] > lm[14][2]
                pinky_down = lm[20][2] > lm[18][2]

                # ===== VOLUME GESTURE =====
                volume_gesture = (
                    thumb_up and index_up and
                    middle_down and ring_down and pinky_down
                )

                thumb = lm[4]
                index = lm[8]

                raw_len = distance(thumb, index)
                smooth_len = alpha * raw_len + (1 - alpha) * prev_length
                prev_length = smooth_len

                vol = np.interp(smooth_len, [MIN_DIST, MAX_DIST], [0, 100])
                vol = int(np.clip(vol, 0, 100))

                if volume_gesture:
                    if abs(vol - last_sent_volume) >= DEAD_ZONE:
                        if OS == "Windows":
                            threading.Thread(
                                target=set_windows_volume,
                                args=(vol,),
                                daemon=True
                            ).start()
                        else:
                            set_volume_async(vol)

                        last_sent_volume = vol

                    # ===== VOLUME VISUALS =====
                    cv2.line(frame, (thumb[1], thumb[2]),
                             (index[1], index[2]), (255, 0, 255), 3)
                    # ===== VOLUME BAR DRAW =====
                    bar_height = np.interp(vol, [0, 100], [BAR_Y_BOTTOM, BAR_Y_TOP])

                    # Background
                    cv2.rectangle(
                        frame,
                        (BAR_X, BAR_Y_TOP),
                        (BAR_X + BAR_WIDTH, BAR_Y_BOTTOM),
                        (60, 60, 60),
                        2
                    )

                    # Filled volume level
                    cv2.rectangle(
                        frame,
                        (BAR_X, int(bar_height)),
                        (BAR_X + BAR_WIDTH, BAR_Y_BOTTOM),
                        (0, 255, 0),
                        -1
                    )

                    # Volume percentage text
                    cv2.putText(
                        frame,
                        f"{vol}%",
                        (BAR_X - 10, BAR_Y_BOTTOM + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )


                # ==================================================
                #   CURSOR MOVE — INDEX ONLY (THUMB DOWN)
                # ==================================================
                if (
                    index_up and
                    not thumb_up and
                    middle_down and ring_down and pinky_down and
                    not volume_gesture
                ):
                    x = np.interp(lm[8][1], [0, w], [0, screen_w])
                    y = np.interp(lm[8][2], [0, h], [0, screen_h])
                    pyautogui.moveTo(x, y, duration=0.04)

                # ==================================================
                #   LEFT CLICK — INDEX + MIDDLE (NO THUMB)
                # ==================================================
                if (
                    index_up and
                    not thumb_up and
                    not middle_down and
                    ring_down and pinky_down and
                    not volume_gesture
                ):
                    if time.time() - last_click_time > CLICK_DELAY:
                        pyautogui.click()
                        last_click_time = time.time()

        cv2.imshow("Hand Gesture Volume Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

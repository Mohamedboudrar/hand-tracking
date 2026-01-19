import cv2
import numpy as np
import platform
import subprocess
import time
import pyautogui
from cvzone.HandTrackingModule import HandDetector

# ================= OS SETUP =================
OS = platform.system()

# ================= HAND SMOOTHING =================
smooth_vol = 0.0
SMOOTHING_ALPHA = 0.15   # 0.1 = smoother | 0.3 = more responsive

# ================= WINDOWS VOLUME STATE =================
current_volume = 50      # tracked software volume (0–100)
VOLUME_STEP = 2          # % per key press (Windows only)

# ================= VOLUME CONTROL =================
def set_volume(vol):
    global current_volume

    vol = int(np.clip(vol, 0, 100))

    # -------- macOS (EXACT) --------
    if OS == "Darwin":
        subprocess.call(
            ["osascript", "-e", f"set volume output volume {vol}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    # -------- Windows (MEDIA KEYS) --------
    elif OS == "Windows":
        delta = vol - current_volume
        if abs(delta) < VOLUME_STEP:
            return

        steps = abs(delta) // VOLUME_STEP
        key = "volumeup" if delta > 0 else "volumedown"

        for _ in range(steps):
            pyautogui.press(key)
            current_volume += VOLUME_STEP if delta > 0 else -VOLUME_STEP

# ================= MAIN LOGIC =================
def main():
    global smooth_vol

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandDetector(detectionCon=0.7, maxHands=1)

    last_click_time = 0
    CLICK_DELAY = 0.5

    screen_w, screen_h = pyautogui.size()

    # Mouse smoothing
    smoothening = 5
    plocX, plocY = 0, 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, draw=True)

        if hands:
            hand = hands[0]
            lmList = hand["lmList"]
            fingers = detector.fingersUp(hand)

            x1, y1 = lmList[4][:2]   # Thumb tip
            x2, y2 = lmList[8][:2]   # Index tip

            # ================= VOLUME MODE =================
            if fingers == [1, 1, 0, 0, 0]:
                length, _, img = detector.findDistance((x1, y1), (x2, y2), img)

                # Raw volume from hand distance
                raw_vol = np.interp(length, [30, 250], [0, 100])
                raw_vol = np.clip(raw_vol, 0, 100)

                # --- Exponential Moving Average (SMOOTHING) ---
                smooth_vol = smooth_vol + SMOOTHING_ALPHA * (raw_vol - smooth_vol)
                vol = int(round(smooth_vol))

                # Volume bar (matches hand exactly)
                bar_height = np.interp(vol, [0, 100], [550, 150])
                cv2.rectangle(img, (1150, 150), (1180, 550), (60, 60, 60), 2)
                cv2.rectangle(img, (1150, int(bar_height)), (1180, 550),
                              (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{vol}%', (1130, 590),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                set_volume(vol)

            # ================= MOUSE MODE =================
            elif fingers == [0, 1, 0, 0, 0]:
                frameR = 150

                x3 = np.interp(x2, (frameR, 1280 - frameR), (0, screen_w))
                y3 = np.interp(y2, (frameR, 720 - frameR), (0, screen_h))

                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                pyautogui.moveTo(clocX, clocY)
                plocX, plocY = clocX, clocY

            # ================= CLICK MODE =================
            if fingers == [0, 1, 0, 0, 1]:
                if time.time() - last_click_time > CLICK_DELAY:
                    pyautogui.click()
                    last_click_time = time.time()
                    cv2.circle(img, (x2, y2), 15, (0, 255, 0), cv2.FILLED)

        cv2.imshow("Hand Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ================= RUN =================
if __name__ == "__main__":
    main()

import cv2
import mediapipe as mp

def main():
    cap = cv2.VideoCapture(0)

    # 1. Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    
    # max_num_hands=1 implies we only track one hand for simplicity
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # 2. Define Finger Landmarks (Indices from MediaPipe documentation)
    # [Thumb, Index, Middle, Ring, Pinky]
    # We look at the TIP (4, 8, 12, 16, 20)
    tip_ids = [4, 8, 12, 16, 20]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)
        
        # MediaPipe works with RGB, OpenCV uses BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image to find hands
        results = hands.process(rgb_frame)

        # List to store which fingers are open (1) or closed (0)
        fingers = []

        # If a hand is found
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # DRAW the skeleton on the hand
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # --- MATCHING LOGIC (Geometry) ---
                # We need to get the coordinate of every landmark
                lm_list = []
                h, w, c = frame.shape
                
                for id, lm in enumerate(hand_landmarks.landmark):
                    # Convert normalized (0-1) coords to pixels (x, y)
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])

                if len(lm_list) != 0:
                    # 1. THUMB LOGIC
                    # The thumb moves horizontally. We check if the tip (4) is 
                    # to the "right" or "left" of the knuckle (3) depending on hand side.
                    # Simple heuristic: Check if tip x is further out than knuckle x.
                    # (Note: This simple logic assumes right hand facing camera)
                    if lm_list[tip_ids[0]][1] < lm_list[tip_ids[0] - 1][1]: 
                        fingers.append(1) # Open
                    else:
                        fingers.append(0) # Closed

                    # 2. FOUR LONG FINGERS LOGIC
                    # For Index(8), Middle(12), Ring(16), Pinky(20):
                    # Check if TIP y < PIP y (y coordinates start 0 at top)
                    for id in range(1, 5):
                        # Tip (ids 8,12,16,20) vs PIP joint (ids 6,10,14,18)
                        # Remember: in image, "Higher" means SMALLER Y value.
                        if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    # --- GESTURE RECOGNITION ---
                    total_fingers = fingers.count(1)
                    
                    gesture = ""
                    if total_fingers == 0:
                        gesture = "Fist"
                    elif total_fingers == 5:
                        gesture = "High Five"
                    elif total_fingers == 2 and fingers[1] == 1 and fingers[2] == 1:
                        gesture = "Peace/Victory"
                    elif total_fingers == 1 and fingers[1] == 1:
                        gesture = "Pointing"
                    else:
                        gesture = f"{total_fingers} Fingers"

                    # Display Result
                    cv2.rectangle(frame, (20, 20), (300, 100), (0, 255, 0), -1)
                    cv2.putText(frame, gesture, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                                2, (255, 255, 255), 3)

        cv2.imshow("MediaPipe Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
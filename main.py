import cv2
import numpy as np
import random
from cvzone.HandTrackingModule import HandDetector
import os

# Settings
window_w, window_h = 1280, 900
dpi = 96
piece_size = int(2 * dpi)  # 2 inches = 192 pixels
rows, cols = 2, 2
image_folder = '.'
image_files = ['puzzle1.png', 'puzzle3.png']

# Initialize
cap = cv2.VideoCapture(0)
cap.set(3, window_w)
cap.set(4, window_h)
detector = HandDetector(detectionCon=0.8)


def load_random_puzzle():
    full_img = None
    while full_img is None:
        img_path = os.path.join(image_folder, random.choice(image_files))
        full_img = cv2.imread(img_path)

    full_img = cv2.resize(full_img, (piece_size, piece_size))
    h, w, _ = full_img.shape
    ph, pw = h // rows, w // cols

    pieces = []
    for r in range(rows):
        for c in range(cols):
            piece_img = full_img[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw]
            pos = (random.randint(50, 400), random.randint(50, 400))
            pieces.append({
                'img': piece_img,
                'target': (c * pw + (window_w - piece_size) // 2, r * ph + 100),
                'pos': pos,
                'locked': False
            })
    return pieces, (h, w)


def draw_ui(img):
    # Title
    cv2.rectangle(img, (0, 0), (window_w, 60), (50, 50, 50), -1)
    cv2.putText(img, "🧩 Hand Puzzle Challenge", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Restart button
    cv2.rectangle(img, (window_w - 160, 15), (window_w - 30, 50), (0, 100, 250), -1)
    cv2.putText(img, "Restart", (window_w - 145, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def inside_button(x, y):
    return window_w - 160 < x < window_w - 30 and 15 < y < 50


# Game state
pieces, puzzle_shape = load_random_puzzle()
selected_piece = None
offset_x, offset_y = 0, 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img)
    draw_ui(img)

    # Puzzle area border
    cv2.rectangle(img, ((window_w - piece_size) // 2, 100), ((window_w + piece_size) // 2, 100 + piece_size),
                  (255, 255, 255), 2)

    if hands:
        hand = hands[0]
        cx, cy = hand['center']
        fingers = detector.fingersUp(hand)

        if fingers[1] and fingers[2]:  # index and middle finger
            if selected_piece is None:
                for i, piece in enumerate(pieces):
                    px, py = piece['pos']
                    ph_, pw_, _ = piece['img'].shape
                    if px < cx < px + pw_ and py < cy < py + ph_ and not piece['locked']:
                        selected_piece = i
                        offset_x = cx - px
                        offset_y = cy - py
                        break
            else:
                pieces[selected_piece]['pos'] = (cx - offset_x, cy - offset_y)
        else:
            if selected_piece is not None:
                piece = pieces[selected_piece]
                x, y = piece['pos']
                tx, ty = piece['target']
                if abs(x - tx) < 30 and abs(y - ty) < 30:
                    piece['pos'] = (tx, ty)
                    piece['locked'] = True
                selected_piece = None

        # Restart button press
        if inside_button(cx, cy) and fingers == [0, 1, 0, 0, 0]:  # only index
            pieces, puzzle_shape = load_random_puzzle()
            selected_piece = None
            continue

    # Draw all pieces
    for piece in pieces:
        x, y = piece['pos']
        ph_, pw_, _ = piece['img'].shape

        # Border always
        cv2.rectangle(img, (x, y), (x + pw_, y + ph_), (200, 200, 200), 2)

        if piece['locked']:
            img[y:y + ph_, x:x + pw_] = piece['img']

    # Completion text
    if all(p['locked'] for p in pieces):
        cv2.putText(img, "✅ Puzzle Complete!", (window_w // 2 - 200, 350), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 0), 4)

    cv2.imshow("Hand Puzzle Game", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

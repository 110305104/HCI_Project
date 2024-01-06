import cv2
import mediapipe as mp
import pygame
import os
import time
import numpy as np

# 初始化MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# 初始化Pygame
pygame.mixer.init()

# 打開攝像頭（0代表預設攝像頭）
cap = cv2.VideoCapture(0)

# 定義熱區的座標（左上角和右下角）以及熱區名稱
hotspots = [
    {"name": "play", "coords": [(50, 50), (200, 150)], "last_triggered": 0},
    {"name": "pause", "coords": [(250, 50), (400, 150)], "last_triggered": 0},
    {"name": "resume", "coords": [(450, 50), (600, 150)], "last_triggered": 0},
    {"name": "hand", "coords": [(50, 350), (200, 450)], "last_triggered": 0},
]

# 載入音樂檔案
script_dir = os.path.dirname(os.path.abspath(__file__))
music_file_path = os.path.join(script_dir, 'Music.mp3')
pygame.mixer.music.load(music_file_path)

#載入背景
background_image_path = os.path.join(script_dir, 'Background.jpg')
try:
    background_image = cv2.imread(background_image_path)
    background_image = cv2.resize(background_image, (640, 480))  # 調整大小
except Exception as e:
    print(f"Error loading or resizing background image: {e}")
    background_image = np.zeros((480, 640, 3), dtype=np.uint8)  # 創建黑色背景作為替代

# 翻轉背景圖
background_image = cv2.flip(background_image, 1)

# 播放音樂
# pygame.mixer.music.play(-1)  # -1 -> 無限循環播放

# 初始化顯示手部的狀態
show_hand = False

# 初始化文字顯示開始時間
text_display_start_time = 0
display_text = ""

# 初始化背景透明度
background_alpha = 0.3  # 初始透明度

while True:
    # 讀取畫面
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # 取消鏡像翻轉
    # 將畫面轉換為RGB格式
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 調整背景透明度
    frame = cv2.addWeighted(frame, 1 - background_alpha, background_image, background_alpha, 0)

    # 檢測手部
    results = hands.process(rgb_frame)


    # 檢測結果處理
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 增加透過手勢調整透明度功能
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # 檢查手指位置
            if index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y and ring_tip.y < thumb_tip.y and pinky_tip.y < thumb_tip.y:
                # "手掌張開，增加背景透明度
                background_alpha = max(0.0, background_alpha - 0.015)
            else:
                # "握拳" 手勢，降低背景透明度
                background_alpha = min(1.0, background_alpha + 0.015)

            # 檢查手是否經過熱區
            for hotspot in hotspots:
                x = int(index_tip.x * frame.shape[1])
                y = int(index_tip.y * frame.shape[0])

                if (hotspot["coords"][0][0] < x < hotspot["coords"][1][0] and
                        hotspot["coords"][0][1] < y < hotspot["coords"][1][1] and
                        time.time() - hotspot["last_triggered"] >= 0.65):  # 檢查冷卻時間
                    hotspot_name = hotspot["name"]

                    # 控制音樂播放狀態
                    if hotspot_name == "play":
                        pygame.mixer.music.play()
                        display_text = "Music Play!"
                    elif hotspot_name == "pause":
                        pygame.mixer.music.pause()
                        display_text = "Music Pause!"
                    elif hotspot_name == "resume":
                        pygame.mixer.music.unpause()
                        display_text = "Music Resume!"
                    elif hotspot_name == "hand":
                        show_hand = not show_hand  # 切換顯示手部的狀態
                        if show_hand:
                            display_text = "Show Hand!"
                        else:
                            display_text = "Not Show Hand!"

                    # 更新最後觸發時間
                    hotspot["last_triggered"] = time.time()

                    # 設定文字顯示開始時間
                    text_display_start_time = time.time()

            # 顯示手部節點和連線（如果 show_hand 為 True）
            landmark_drawing_spec_point = mp_drawing.DrawingSpec(color=(102,204,0))
            landmark_drawing_spec_line = mp_drawing.DrawingSpec(color=font_color)
            if show_hand:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec_point, landmark_drawing_spec_line)



    # 繪製熱區
    # 動態計算熱區邊框和熱區字體的顏色
    font_color = tuple(int(c * (1-background_alpha)) for c in (255, 255, 255))
    # font_color = (0,0,0)
    # 顯示熱區邊框    
    for hotspot in hotspots:
        cv2.rectangle(frame, hotspot["coords"][0], hotspot["coords"][1], font_color, 2)

    # 顯示熱區名字
    for hotspot in hotspots:
        text_position = (hotspot["coords"][0][0], hotspot["coords"][0][1] - 10)
        cv2.putText(frame, hotspot["name"], text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)

    # 顯示文字
    if time.time() - text_display_start_time <= 1.5:
        # 顯示觸發指令在畫面正中間
        text_size_hand = cv2.getTextSize(f'Hand triggered {hotspot_name}', cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x_hand = int((frame.shape[1] - text_size_hand[0]) / 2)
        cv2.putText(frame, f'Hand triggered {hotspot_name}', (text_x_hand, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    font_color, 2)

        text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = int((frame.shape[1] - text_size[0]) / 2)
        text_y = int((frame.shape[0] + text_size[1]) / 2)+20
        cv2.putText(frame, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    font_color, 2)

    # 顯示畫面
    cv2.imshow('Camera', frame)

    # 關閉
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q') or cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:  # 如果按下 "esc" 鍵或 "q" 鍵，或點關閉視窗，退出迴圈
        break

# 釋放攝像頭資源
cap.release()

# 關閉所有視窗
cv2.destroyAllWindows()

import cv2
import numpy as np
import time
import pygame
import os

# 啟動攝像頭
cap = cv2.VideoCapture(0)

# 初始化 Pygame 音效
pygame.mixer.init()

# 設定不同指令的 ROI (Region of Interest) 區域
play_roi = (50, 50, 150, 150)
pause_roi = (250, 50, 150, 150)
resume_roi = (450, 50, 150, 150)

# 背景模型初始化
background_models = {
    'play': None,
    'pause': None,
    'resume': None
}

# 累積變化量初始化
accumulated_changes = {
    'play': 0,
    'pause': 0,
    'resume': 0
}

# 觸發閾值設定
thresholds = {
    'play': 60000,
    'pause': 60000,
    'resume': 60000
}

# 冷卻時間設定
cooldown_time = 1.5

# 最後觸發時間初始化
last_triggered_time = {
    'play': 0,
    'pause': 0,
    'resume': 0
}

# 設定檔案路徑
script_dir = os.path.dirname(os.path.abspath(__file__))
music_file_path = os.path.join(script_dir, 'Music.mp3')
pygame.mixer.music.load(music_file_path)
background_image_path = os.path.join(script_dir, 'Background.jpg')

try:
    # 讀取並調整背景圖片大小
    background_image = cv2.imread(background_image_path)
    background_image = cv2.resize(background_image, (640, 480))  
except Exception as e:
    # 背景圖片讀取或調整大小出錯時的處理
    print(f"Error loading or resizing background image: {e}")
    background_image = np.zeros((480, 640, 3), dtype=np.uint8)  # 創建黑色背景作為替代

# 顯示文字和時間初始化
display_text = ""
display_start_time = 0

# 主迴圈
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # 取消鏡像翻轉
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 將背景圖片疊加在畫面上
    frame = cv2.addWeighted(frame, 0.3, background_image, 0.8, 0)

    # 在每一幀的開始處將變化量重置為零（避免細微變動不斷累加觸發指令）
    for roi_name in ['play', 'pause', 'resume']:
        accumulated_changes[roi_name] = 0

    # 取得當前時間
    current_time = time.time()

    # 檢查每個 ROI
    for roi_name, roi_params in [('play', play_roi), ('pause', pause_roi), ('resume', resume_roi)]:
        roi_x, roi_y, roi_width, roi_height = roi_params
        roi = gray[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        if background_models[roi_name] is None:
            background_models[roi_name] = roi.copy().astype("float")
            continue

        cv2.accumulateWeighted(roi, background_models[roi_name], 0.5)
        delta = cv2.absdiff(roi, cv2.convertScaleAbs(background_models[roi_name]))

        accumulated_changes[roi_name] += np.sum(delta)

        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)

        # 在熱區上顯示指令
        cv2.putText(frame, f'{roi_name.capitalize()} - Change: {accumulated_changes[roi_name]}', (roi_x, roi_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 檢查指令觸發的冷卻時間
        if accumulated_changes[roi_name] > thresholds[roi_name] and current_time - last_triggered_time[roi_name] > cooldown_time:
            print(f"Triggered {roi_name} command!")
            if roi_name == 'play':
                pygame.mixer.music.play()
                display_text = "Music start"
            elif roi_name == 'pause':
                pygame.mixer.music.pause()
                display_text = "Music paused"
            elif roi_name == 'resume':
                pygame.mixer.music.unpause()
                display_text = "Music resumed"

            accumulated_changes[roi_name] = 0
            last_triggered_time[roi_name] = current_time
            display_start_time = current_time

    # 在畫面中央顯示指令觸發文字
    if current_time - display_start_time < 1:
        cv2.putText(frame, display_text, (int(frame.shape[1] / 2) - int(len(display_text) * 5), int(frame.shape[0] / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 顯示畫面
    cv2.imshow('Camera Feed', frame)

    # 按下 'q'、'esc' 鍵或視窗關閉按鈕可關閉程式
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27 or cv2.getWindowProperty('Camera Feed', cv2.WND_PROP_VISIBLE) < 1:
        break

# 釋放攝像頭資源
cap.release()

# 關閉所有視窗
cv2.destroyAllWindows()

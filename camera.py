import cv2
import numpy as np
from ultralytics import YOLO

# 1. セグメンテーション用モデルの読み込み
model = YOLO("yolov8s-seg.pt") 

# 2. 画像の読み込み
image_path = "unnamed.jpg"
frame = cv2.imread(image_path)
if frame is None:
    print(f"❌ エラー: {image_path} が見つかりません。")
    exit()

h, w, _ = frame.shape

# 3. AIで「人」の領域を大まかに抽出
results = model(frame, classes=[0], verbose=False)

# 人がいる部分を白（255）、それ以外を黒（0）にするマスク画像を作成
person_mask = np.zeros((h, w), dtype=np.uint8)

if results[0].masks is not None:
    for mask_data in results[0].masks.xy:
        pts = mask_data.astype(np.int32)
        # AIが「人」と反応した部分を白い多角形で塗りつぶす
        cv2.fillPoly(person_mask, [pts], 255)

# 4. 【ここがポイント】「人の気配」をにじませて、密集した塊（エリア）を作る
# 大きめのカーネルでぼかしと膨張処理を行い、点在する人を一つの「エリア」に結合します
kernel_blur = 51
person_blur = cv2.GaussianBlur(person_mask, (kernel_blur, kernel_blur), 0)
_, crowd_area = cv2.threshold(person_blur, 10, 255, cv2.THRESH_BINARY)

# 5. 自動生成された密集エリアの輪郭を抽出
contours, _ = cv2.findContours(crowd_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 密集エリアの総ピクセル数を計算
total_crowd_pixels = np.sum(crowd_area == 255)
# 画面全体に対する「人が集まっているエリア」の占有率（簡易的な密度スコア）
density_score = (total_crowd_pixels / (w * h)) * 100

# 6. 元の画像に「AIが自動で囲んだ群衆エリア」を緑線で描画
cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

# 検出結果を画面に表示
cv2.putText(frame, f"Crowd Area Auto-Detected!", (30, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
cv2.putText(frame, f"Space Occupancy: {density_score:.1f}%", (30, 110), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

# 7. ウィンドウ表示
cv2.imshow("AI Auto Crowd Detection (Green Boundary)", frame)
cv2.imshow("Generated Crowd Mask (Area)", crowd_area)
cv2.waitKey(0)
cv2.destroyAllWindows()
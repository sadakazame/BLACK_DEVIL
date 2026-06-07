import datetime
import time
import cv2
import numpy as np
from ultralytics import YOLO

# ==============================================================================
# 設定・初期化
# ==============================================================================
# YOLOv8モデルの読み込み（初回のみ自動ダウンロードされます）
model = YOLO("yolov8n.pt")

# カメラの設定（ユーザー様の指定通り「1」番のカメラを開きます）
# ※本番でIPカメラを使う場合は、RTSPのURL "rtsp://..." に書き換えてください
cap = cv2.VideoCapture(0)

# 明るさ判定のしきい値（80未満を夜間モードと判定）
BRIGHTNESS_THRESHOLD = 80

# 何秒ごとに判定・データ出力するかを設定（10秒おき）
INTERVAL_SECONDS = 30

# 最後に処理した時間を記録する変数
last_processed_time = 0


def preprocess_frame(frame):
    """フレームの明るさを判定し、必要に応じて夜間用の前処理（CLAHE）を行う関数"""
    # HSV色空間に変換して、輝度（V）チャンネルを取得
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    # 画面全体の平均輝度を計算
    avg_brightness = np.mean(v_channel)

    is_night = False
    if avg_brightness < BRIGHTNESS_THRESHOLD:
        is_night = True
        # 夜間判定：コントラスト有限適応ヒストグラム平滑化（CLAHE）を適用
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(v_channel)
        # BGR（通常のカラー画像）に戻す
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 軽めのノイズ除去（ガウシアンフィルタ）
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

    return frame, avg_brightness, is_night


# ==============================================================================
# メインループ
# ==============================================================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("カメラから映像を取得できない、または動画の終端です。")
        break

    # 現在の時刻（秒単位）を取得
    current_time = time.time()

    # ★前回の処理から「指定した秒数（10秒）」が経過しているかチェック
    if current_time - last_processed_time >= INTERVAL_SECONDS:
        # 条件を満たしたら、最後に処理した時間を「今」に更新
        last_processed_time = current_time

        # ----------------------------------------------------------------------
        # 【重要】10秒に1回だけ実行されるエリア（インデントを下げています）
        # ----------------------------------------------------------------------

        # 1. 昼夜判定と前処理
        processed_frame, brightness, is_night = preprocess_frame(frame)

        # 2. YOLOによる物体検出（「人」だけを検出）
        results = model(processed_frame, classes=[0], conf=0.25, verbose=False)

        # 3. 他チームへ渡すためのデータ構造（JSONベース）の作成
        detection_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "is_night_mode": is_night,
            "avg_brightness": round(brightness, 2),
            "total_count": 0,
            "people": [],
        }

        # 検出結果のパース
        if len(results) > 0:
            boxes = results[0].boxes
            detection_data["total_count"] = len(boxes)

            for i, box in enumerate(boxes):
                # 座標の取得と整数への変換
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)

                # 幅と高さ、および後続チームが計算しやすい「中心座標」の計算
                width = x2 - x1
                height = y2 - y1
                center_x = int(x1 + (width / 2))
                center_y = int(y1 + (height / 2))

                # 確信度
                confidence = float(box.conf[0])

                # 各個人のデータを格納
                person_info = {
                    "id": i + 1,
                    "center_x": center_x,
                    "center_y": center_y,
                    "width": width,
                    "height": height,
                    "confidence": round(confidence, 2),
                }
                detection_data["people"].append(person_info)

                # 【デバッグ用】画面への描画（夜は赤、昼は緑）
                color = (0, 0, 255) if is_night else (0, 255, 0)
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(processed_frame, (center_x, center_y), 4, color, -1)

        # データのコンソール出力（他チームへの連携部分）
        print(
            f"Time: {detection_data['timestamp']} | Mode: {'NIGHT' if is_night else 'DAY'} | Count: {detection_data['total_count']}"
        )
        # 後続チームへJSONを渡す際は、ここで送出ロジックを書きます
        # 例：send_to_backend(json.dumps(detection_data))

        # 【デバッグ用】画面表示への文字入れ
        cv2.putText(
            processed_frame,
            f"Mode: {'Night' if is_night else 'Day'} ({brightness:.1f})",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            processed_frame,
            f"Count: {detection_data['total_count']}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # プレビューウィンドウの更新
        cv2.imshow("YOLO Detection (Top-down)", processed_frame)

    # ----------------------------------------------------------------------
    # 10秒の制限の外側（毎フレーム実行）
    # ----------------------------------------------------------------------
    # 'q' キーが押されたら安全に終了する（ウィンドウのフリーズを防ぐため外に配置）
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 後片付け
cap.release()
cv2.destroyAllWindows()
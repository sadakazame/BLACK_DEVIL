import datetime
import time
import cv2
import numpy as np
from ultralytics import YOLO

# --- グローバル変数（マウス操作・エリア設定用） ---
grid_state = None
cell_w = 0
cell_h = 0
drawing = False

# ==============================================================================
# 1. エリア設定用の関数（eria.py より移植・最適化）
# ==============================================================================
def select_grid_callback(event, x, y, flags, param):
    """マウスクリックおよびドラッグでグリッドのON/OFFを切り替えるコールバック関数"""
    global grid_state, cell_w, cell_h, drawing

    col = x // cell_w
    row = y // cell_h

    if row >= grid_state.shape[0] or col >= grid_state.shape[1]:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        grid_state[row, col] = not grid_state[row, col]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            grid_state[row, col] = True

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


def setup_fence_area(frame, rows=12, cols=16):
    """初期起動時にユーザーがグリッドで監視エリアを選択し、マスク画像を生成する"""
    global grid_state, cell_w, cell_h

    h, w = frame.shape[:2]
    cell_w = w // cols
    cell_h = h // rows

    grid_state = np.zeros((rows, cols), dtype=bool)

    window_name = "Select Fence Area (Press ENTER to confirm)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, select_grid_callback)

    print("\n" + "=" * 50)
    print("【エリア設定モード】")
    print(" マウスでクリックまたはドラッグして、監視したいエリアを選択してください。")
    print(" 選択が完了したら 'Enter' または 'Space' キーを押してください。")
    print("=" * 50 + "\n")

    while True:
        display = frame.copy()
        overlay = frame.copy()

        # 選択されたセルを赤色（半透明用）にする
        for r in range(rows):
            for c in range(cols):
                if grid_state[r, c]:
                    pt1 = (c * cell_w, r * cell_h)
                    pt2 = ((c + 1) * cell_w, (r + 1) * cell_h)
                    cv2.rectangle(overlay, pt1, pt2, (0, 0, 255), -1)

        cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)

        # グリッド線の描画
        for r in range(1, rows):
            cv2.line(display, (0, r * cell_h), (w, r * cell_h), (255, 255, 255), 1)
        for c in range(1, cols):
            cv2.line(display, (c * cell_w, 0), (c * cell_w, h), (255, 255, 255), 1)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 32:  # Enter or Space
            break

    cv2.destroyWindow(window_name)

    # 白黒のマスク画像を生成 (エリア内: 255, エリア外: 0)
    final_mask = np.zeros((h, w), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if grid_state[r, c]:
                pt1 = (c * cell_w, r * cell_h)
                pt2 = ((c + 1) * cell_w, (r + 1) * cell_h)
                cv2.rectangle(final_mask, pt1, pt2, 255, -1)

    return final_mask


# ==============================================================================
# 2. 画像の前処理（camera.py より移植）
# ==============================================================================
def preprocess_frame(frame, brightness_threshold=80):
    """明るさを判定し、必要に応じて夜間用の前処理（CLAHE）を行う"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    avg_brightness = np.mean(v_channel)

    is_night = False
    if avg_brightness < brightness_threshold:
        is_night = True
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(v_channel)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

    return frame, avg_brightness, is_night


# ==============================================================================
# 3. メイン処理
# ==============================================================================
def main():
    # モデルとカメラの初期化
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)  # 環境に合わせて変更してください

    if not cap.isOpened():
        print("【エラー】カメラにアクセスできませんでした。")
        return

    # カメラの映像安定化
    for _ in range(5):
        cap.read()

    ret, initial_frame = cap.read()
    if not ret:
        print("初期フレームの取得に失敗しました。")
        return

    # --- [STEP 1] 初期セットアップ：エリア設定マスクの作成 ---
    # rows, cols でグリッドの細かさを調整可能
    fence_mask = setup_fence_area(initial_frame, rows=12, cols=16)

    # 設定用の中間の反転マスク（エリア外を暗くする視覚効果用）
    inverse_mask = cv2.bitwise_not(fence_mask)

    print("エリアの設定が完了しました。監視を開始します...")
    print("「q」キーを押すと終了します。")

    # タイマー用の変数（元のコードの30秒設定を維持）
    INTERVAL_SECONDS = 30
    last_processed_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("カメラから映像を取得できない、または動画の終端です。")
            break

        current_time = time.time()

        # 指定したインターバル（30秒）ごとにYOLO検出とデータ出力を実行
        if current_time - last_processed_time >= INTERVAL_SECONDS:
            last_processed_time = current_time

            # 昼夜判定と画像前処理
            processed_frame, brightness, is_night = preprocess_frame(frame)

            # YOLOによる物体検出（「人: classes=[0]」だけを検出）
            results = model(processed_frame, classes=[0], conf=0.25, verbose=False)

            # JSON用データ構造の初期化
            detection_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "is_night_mode": is_night,
                "avg_brightness": round(brightness, 2),
                "total_count": 0,  # エリア内の合計人数
                "people": [],
            }

            if len(results) > 0:
                boxes = results[0].boxes
                area_inside_count = 0

                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, xyxy)

                    width = x2 - x1
                    height = y2 - y1
                    center_x = int(x1 + (width / 2))
                    center_y = int(y1 + (height / 2))

                    # 画面外エラーを防ぐガード
                    h_max, w_max = fence_mask.shape[:2]
                    check_x = min(max(0, center_x), w_max - 1)
                    check_y = min(max(0, center_y), h_max - 1)

                    # ★【重要】人の中心点が、設定したマスク内（値が255）にあるか判定
                    if fence_mask[check_y, check_x] == 255:
                        area_inside_count += 1
                        confidence = float(box.conf[0])

                        # エリア内の人のみリストに追加
                        person_info = {
                            "id": area_inside_count,
                            "center_x": center_x,
                            "center_y": center_y,
                            "width": width,
                            "height": height,
                            "confidence": round(confidence, 2),
                        }
                        detection_data["people"].append(person_info)

                        # 【画面描画】エリア内の人だけ描画（夜は赤、昼は緑）
                        color = (0, 0, 255) if is_night else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.circle(frame, (center_x, center_y), 4, color, -1)

                # カウントの更新
                detection_data["total_count"] = area_inside_count

            # コンソールへのデータ出力
            print(
                f"Time: {detection_data['timestamp']} | Mode: {'NIGHT' if is_night else 'DAY'} | Area Count: {detection_data['total_count']}"
            )

            # --- 視覚効果：エリア外を少し暗くして画面に文字を合成 ---
            dark_frame = (frame * 0.3).astype(np.uint8)
            highlighted_area = cv2.bitwise_and(frame, frame, mask=fence_mask)
            darkened_area = cv2.bitwise_and(dark_frame, dark_frame, mask=inverse_mask)
            monitoring_display = cv2.add(highlighted_area, darkened_area)

            cv2.putText(
                monitoring_display,
                f"Mode: {'Night' if is_night else 'Day'} ({brightness:.1f})",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                monitoring_display,
                f"Area Count: {detection_data['total_count']}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # プレビューウィンドウの更新
            cv2.imshow("Fence Area Monitoring", monitoring_display)

        # 毎フレーム実行（終了チェック用）
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("監視を終了しました。")


if __name__ == "__main__":
    main()
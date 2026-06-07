import datetime
import json
import cv2
import numpy as np
from ultralytics import YOLO

# --- グローバル変数 ---
grid_state = None
cell_w = 0
cell_h = 0
drawing = False

# ==============================================================================
# 1. 設定・初期化
# ==============================================================================
IMAGE_PATH = "noonlot.png"

# さらに検出能力が高い大型モデルに変更（初回のみ自動ダウンロードされます）
model = YOLO("yolov8x.pt") 
BRIGHTNESS_THRESHOLD = 80


# ==============================================================================
# 2. エリア設定用の関数
# ==============================================================================
def select_grid_callback(event, x, y, flags, param):
    global grid_state, cell_w, cell_h, drawing
    col = int(x // cell_w)
    row = int(y // cell_h)
    if row >= grid_state.shape[0] or col >= grid_state.shape[1] or row < 0 or col < 0:
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
    global grid_state, cell_w, cell_h

    h, w = frame.shape[:2]
    cell_w = w / cols
    cell_h = h / rows
    grid_state = np.zeros((rows, cols), dtype=bool)

    window_name = "Select Fence Area"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, select_grid_callback)

    print("\n" + "=" * 50)
    print("【エリア設定モード】")
    print(" 人がいるエリアをマウスで赤く染めてください（Enterで確定）。")
    print("=" * 50 + "\n")

    while True:
        display = frame.copy()
        overlay = frame.copy()

        for r in range(rows):
            for c in range(cols):
                if grid_state[r, c]:
                    pt1 = (int(c * cell_w), int(r * cell_h))
                    pt2 = (int((c + 1) * cell_w), int((r + 1) * cell_h))
                    cv2.rectangle(overlay, pt1, pt2, (0, 0, 255), -1)

        cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)

        for r in range(1, rows):
            cv2.line(display, (0, int(r * cell_h)), (w, int(r * cell_h)), (255, 255, 255), 1)
        for c in range(1, cols):
            cv2.line(display, (int(c * cell_w), 0), (int(c * cell_w), h), (255, 255, 255), 1)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF
        if key == 13 or key == 32:
            break

    cv2.destroyWindow(window_name)
    return grid_state


def preprocess_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    avg_brightness = np.mean(v_channel)

    is_night = False
    if avg_brightness < BRIGHTNESS_THRESHOLD:
        is_night = True
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(v_channel)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

    return frame, avg_brightness, is_night


# ==============================================================================
# 4. メイン処理
# ==============================================================================
def main():
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print(f"【エラー】画像 '{IMAGE_PATH}' が読み込めませんでした。")
        return

    processed_frame, brightness, is_night = preprocess_frame(frame)

    # ユーザー選択
    BASE_ROWS, BASE_COLS = 12, 16
    rough_grid = setup_fence_area(frame, rows=BASE_ROWS, cols=BASE_COLS)

    # さらに細かいスキャン（親グリッドの3倍：36x48マスに細分化）
    SCAN_ROWS, SCAN_COLS = BASE_ROWS * 3, BASE_COLS * 3
    
    h, w = frame.shape[:2]
    c_w = w / SCAN_COLS
    c_h = h / SCAN_ROWS

    fence_mask = np.zeros((h, w), dtype=np.uint8)
    for r in range(SCAN_ROWS):
        for c in range(SCAN_COLS):
            if rough_grid[r // 3, c // 3]:
                pt1 = (int(c * c_w), int(r * c_h))
                pt2 = (int((c + 1) * c_w), int((r + 1) * c_h))
                cv2.rectangle(fence_mask, pt1, pt2, 255, -1)

    inverse_mask = cv2.bitwise_not(fence_mask)
    print("[INFO] 高解像度拡大 ＋ 超低しきい値スキャンを実行中。少し時間がかかります...")
    
    all_boxes = []
    
    for r in range(SCAN_ROWS):
        for c in range(SCAN_COLS):
            if rough_grid[r // 3, c // 3]:
                buffer = 20
                x_start = max(0, int(c * c_w) - buffer)
                y_start = max(0, int(r * c_h) - buffer)
                x_end = min(w, int((c + 1) * c_w) + buffer)
                y_end = min(h, int((r + 1) * c_h) + buffer)
                
                grid_crop = processed_frame[y_start:y_end, x_start:x_end]
                
                if grid_crop.size == 0:
                    continue
                
                # ★【超重要】切り出した極小のマスを、AIが認識しやすいように「さらに3倍に強制拡大」する
                grid_zoomed = cv2.resize(grid_crop, (grid_crop.shape[1] * 3, grid_crop.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
                
                # ★【超重要】しきい値を限界（0.05）まで下げ、検出サイズ制限を解除（imgsz=1280）
                results = model(grid_zoomed, classes=[0], conf=0.05, imgsz=1280, verbose=False)
                
                if len(results) > 0 and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        
                        # 3倍拡大した座標から元の切り出し座標へ逆算
                        crop_x1 = xyxy[0] / 3
                        crop_y1 = xyxy[1] / 3
                        crop_x2 = xyxy[2] / 3
                        crop_y2 = xyxy[3] / 3
                        
                        # 全体座標に復元
                        global_x1 = crop_x1 + x_start
                        global_y1 = crop_y1 + y_start
                        global_x2 = crop_x2 + x_start
                        global_y2 = crop_y2 + y_start
                        all_boxes.append([global_x1, global_y1, global_x2, global_y2, box.conf[0].cpu().numpy()])

    # 重複ボックスの統合（NMS）※重なり許容度を少し上げて、より細かく残す
    final_people_boxes = []
    if len(all_boxes) > 0:
        all_boxes = np.array(all_boxes)
        nms_indices = cv2.dnn.NMSBoxes(
            all_boxes[:, :4].astype(int).tolist(), 
            all_boxes[:, 4].astype(float).tolist(), 
            score_threshold=0.05, 
            nms_threshold=0.5
        )
        if len(nms_indices) > 0:
            for i in nms_indices.flatten():
                final_people_boxes.append(all_boxes[i])

    detection_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "is_night_mode": is_night,
        "avg_brightness": round(brightness, 2),
        "total_count": 0,
        "people": [],
    }

    output_image = frame.copy()
    area_inside_count = 0

    for box in final_people_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        width = x2 - x1
        height = y2 - y1
        center_x = int(x1 + (width / 2))
        center_y = int(y1 + (height / 2))

        check_x = min(max(0, center_x), w - 1)
        check_y = min(max(0, center_y), h - 1)

        if fence_mask[check_y, check_x] == 255:
            area_inside_count += 1
            confidence = float(box[4])

            person_info = {
                "id": area_inside_count,
                "center_x": center_x,
                "center_y": center_y,
                "width": width,
                "height": height,
                "confidence": round(confidence, 2),
            }
            detection_data["people"].append(person_info)

            # 最終的な結果描画（小さめの点と薄めの枠で描画）
            color = (0, 0, 255) if is_night else (0, 255, 0)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 1)
            cv2.circle(output_image, (center_x, center_y), 2, color, -1)

    detection_data["total_count"] = area_inside_count

    print("\n=== 超精密群衆スキャン完了後の JSON データ ===")
    print(json.dumps(detection_data, indent=2, ensure_ascii=False))

    with open("verification_result.json", "w", encoding="utf-8") as f:
        json.dump(detection_data, f, indent=2, ensure_ascii=False)

    dark_frame = (output_image * 0.3).astype(np.uint8)
    highlighted_area = cv2.bitwise_and(output_image, output_image, mask=fence_mask)
    darkened_area = cv2.bitwise_and(dark_frame, dark_frame, mask=inverse_mask)
    monitoring_display = cv2.add(highlighted_area, darkened_area)

    cv2.putText(monitoring_display, f"Mode: {'Night' if is_night else 'Day'} ({brightness:.1f})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(monitoring_display, f"Area Count: {detection_data['total_count']}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    screen_h, screen_w = 720, 1280
    scale = min(screen_w / w, screen_h / h)
    show_frame = cv2.resize(monitoring_display, (int(w * scale), int(h * scale))) if scale < 1.0 else monitoring_display

    cv2.imshow("YOLO Crowd Deep Scan", show_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
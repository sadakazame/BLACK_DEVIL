
import datetime
import json
import cv2
import numpy as np
from ultralytics import YOLO

# ==============================================================================
# 1. 設定・初期化
# ==============================================================================
# 【ここを変更】検証したい画像ファイルの名前
IMAGE_PATH = "PXL_20260525_070002046.jpg"

# カメラ版と共通のYOLOv8軽量モデル
model = YOLO("yolov8n.pt")

# カメラ版と共通の明るさしきい値（80未満で夜間モード起動）
BRIGHTNESS_THRESHOLD = 80


# ==============================================================================
# 2. 前処理関数（カメラ版と完全に同一のロジック）
# ==============================================================================
def preprocess_frame(frame):
    """画像の明るさを判定し、必要に応じて夜間用の前処理（CLAHE）を行う関数"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    # 平均輝度の計算
    avg_brightness = np.mean(v_channel)

    is_night = False
    if avg_brightness < BRIGHTNESS_THRESHOLD:
        is_night = True
        # コントラスト有限適応ヒストグラム平滑化（CLAHE）
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(v_channel)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # ノイズ除去
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

    return frame, avg_brightness, is_night


# ==============================================================================
# 3. メイン処理（1枚の画像に対する検証フェーズ）
# ==============================================================================
def main():
    # 画像の読み込み
    frame = cv2.imread(IMAGE_PATH)

    if frame is None:
        print(
            f"【エラー】画像 '{IMAGE_PATH}' が読み込めませんでした。"
        )
        print("ファイル名が正しいか、このPythonファイルと同じフォルダにあるか確認してください。")
        return

    # 1. 昼夜判定と前処理の実行
    processed_frame, brightness, is_night = preprocess_frame(frame)

    # 2. YOLOv8による人間(class 0)の検出
    results = model(processed_frame, classes=[0], conf=0.25, verbose=False)

    # 3. 他チーム連携用データ構造（JSONベース）の作成
    detection_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "is_night_mode": is_night,
        "avg_brightness": round(brightness, 2),
        "total_count": 0,
        "people": [],
    }

    # 4. 検出結果の解析と画面への描画処理
    if len(results) > 0:
        boxes = results[0].boxes
        detection_data["total_count"] = len(boxes)

        for i, box in enumerate(boxes):
            # 座標の取得
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)

            # 幅・高さ・中心座標の計算（他チーム要求仕様）
            width = x2 - x1
            height = y2 - y1
            center_x = int(x1 + (width / 2))
            center_y = int(y1 + (height / 2))

            # 確信度
            confidence = float(box.conf[0])

            # 各個人の詳細データを辞書に追加
            person_info = {
                "id": i + 1,
                "center_x": center_x,
                "center_y": center_y,
                "width": width,
                "height": height,
                "confidence": round(confidence, 2),
            }
            detection_data["people"].append(person_info)

            # プレビュー用描画（夜間モードは赤、通常は緑）
            color = (0, 0, 255) if is_night else (0, 255, 0)
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(processed_frame, (center_x, center_y), 4, color, -1)

    # ==========================================================================
    # 4. 結果の出力とファイル保存
    # ==========================================================================
    # ターミナルへのJSON出力（データの型が正しいかチェック用）
    print("\n=== [DEBUG] 生成された JSON データ ===")
    print(json.dumps(detection_data, indent=2, ensure_ascii=False))

    # サンプルファイル（.json）としてPCに保存
    output_filename = "verification_result.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(detection_data, f, indent=2, ensure_ascii=False)
    print(
        f"\n[INFO] データを '{output_filename}' に保存しました。他チームへの共有に使えます。"
    )

    # ==========================================================================
    # 5. プレビュー表示
    # ==========================================================================
    # 画面に判定ステータスをテキスト表示
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

    print("\n[INFO] プレビュー画面を開きます。終了するには画像ウィンドウ上で何かキーを押してください。")
    cv2.imshow("YOLO Verification (Static Image)", processed_frame)

    # キー入力を無限に待機（画面を閉じずに固定するため、引数は0）
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
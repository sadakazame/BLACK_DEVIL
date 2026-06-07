import time
import cv2
import numpy as np
import serial  # シリアル通信用のライブラリ
import streamlit as st
from ultralytics import YOLO

# ページの設定
st.set_page_config(page_title="壁内人口密度監視システム", layout="wide")

# ==============================================================================
# 0. マイコン（シリアルポート）の接続設定
# ==============================================================================
SERIAL_PORT = "COM7"  # デバイスマネージャーで確認したポート
BAUD_RATE = 38400  # マイコン側の設定（huart2.Init.BaudRate = 38400）

@st.cache_resource
def init_models_and_serial():
    """重いAIモデルのロードとシリアル接続を1度だけキャッシュ化します"""
    # 速度と検出力のバランスが良いMediumモデルを使用
    model = YOLO("yolov8m.pt")
    
    ser = None
    try:
        if 'ser' in globals() and ser is not None and ser.is_open:
            return model, ser
            
        new_ser = serial.Serial()
        new_ser.port = SERIAL_PORT
        new_ser.baudrate = BAUD_RATE
        new_ser.timeout = 1
        new_ser.open()
        time.sleep(2)  # 接続安定のためのウェイト
        return model, new_ser
    except Exception as e:
        print(f"\n[⚠️ SERIAL ERROR] マイコン接続失敗詳細: {e}\n")
        st.sidebar.warning(f"マイコン ({SERIAL_PORT}) が未接続です。実機なしモードで起動します。")
        return model, None

model, ser = init_models_and_serial()

# ==============================================================================
# 1. 設定・状態管理
# ==============================================================================
IMAGE_PATH = "noonlot.png"  # 使用する画像ファイル名
BRIGHTNESS_THRESHOLD = 80

if "last_hardware_state" not in st.session_state:
    st.session_state.last_hardware_state = False
if "fence_mask" not in st.session_state:
    st.session_state.fence_mask = None
if "rough_grid" not in st.session_state:
    st.session_state.rough_grid = None
if "final_people_boxes" not in st.session_state:
    st.session_state.final_people_boxes = None

# コールバック用の一時変数
grid_state_ui = None
cell_w_ui = 0
cell_h_ui = 0
drawing_ui = False

# ==============================================================================
# 2. エリア設定用の関数（OpenCVポップアップ）
# ==============================================================================
def select_grid_callback(event, x, y, flags, param):
    global grid_state_ui, cell_w_ui, cell_h_ui, drawing_ui
    col = int(x // cell_w_ui)
    row = int(y // cell_h_ui)
    if row >= grid_state_ui.shape[0] or col >= grid_state_ui.shape[1] or row < 0 or col < 0:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_ui = True
        grid_state_ui[row, col] = not grid_state_ui[row, col]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_ui:
            grid_state_ui[row, col] = True
    elif event == cv2.EVENT_LBUTTONUP:
        drawing_ui = False


def setup_fence_area_popup(frame, rows=12, cols=16):
    global grid_state_ui, cell_w_ui, cell_h_ui, drawing_ui
    h, w = frame.shape[:2]
    cell_w_ui = w / cols
    cell_h_ui = h / rows
    grid_state_ui = np.zeros((rows, cols), dtype=bool)
    drawing_ui = False

    window_name = "SETUP: Select Area (Press ENTER to Confirm)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, select_grid_callback)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    while True:
        display = frame.copy()
        overlay = frame.copy()
        for r in range(rows):
            for c in range(cols):
                if grid_state_ui[r, c]:
                    pt1 = (int(c * cell_w_ui), int(r * cell_h_ui))
                    pt2 = (int((c + 1) * cell_w_ui), int((r + 1) * cell_h_ui))
                    cv2.rectangle(overlay, pt1, pt2, (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
        for r in range(1, rows):
            cv2.line(display, (0, int(r * cell_h_ui)), (w, int(r * cell_h_ui)), (255, 255, 255), 1)
        for c in range(1, cols):
            cv2.line(display, (int(c * cell_w_ui), 0), (int(c * cell_w_ui), h), (255, 255, 255), 1)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(30) & 0xFF
        if key == 13 or key == 32: break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break
    cv2.destroyWindow(window_name)
    return grid_state_ui


# ==============================================================================
# 3. 前処理関数
# ==============================================================================
def preprocess_frame(frame, threshold=80):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    avg_brightness = np.mean(v_channel)
    is_night = False
    if avg_brightness < threshold:
        is_night = True
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(v_channel)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
    return frame, avg_brightness, is_night


# ==============================================================================
# 4. あなたの担当：マイコン制御送信ロジック（実機連動）
# ==============================================================================
def control_hardware_real(activate: bool, current_density: float):
    """実際のマイコンへ信号を送信し、同時にコンソールにもログを残す関数"""
    if activate and not st.session_state.last_hardware_state:
        print(f"\n[🔴 HARDWARE] 警告レベル到達 ({current_density:.1f}%) -> マイコンに 'H' 送信")
        if ser and ser.is_open:
            ser.write(b"H")  # 点灯命令を送信
        st.session_state.last_hardware_state = True
        
    elif not activate and st.session_state.last_hardware_state:
        print(f"\n[🟢 HARDWARE] 安全レベル復帰 ({current_density:.1f}%) -> マイコンに 'L' 送信")
        if ser and ser.is_open:
            ser.write(b"L")  # 消灯命令を送信
        st.session_state.last_hardware_state = False


# ==============================================================================
# 5. Streamlit UI のレイアウト構築
# ==============================================================================
st.title("🛡️ 壁内人口密度監視システム (精密AIズーム・マイコン連動版)")

st.sidebar.header("⚙️ システム設定")
run_system = st.sidebar.checkbox("システムを起動（ループ開始）", value=False)

# 分母（基準人数）を150人などに広げられるよう最大値を拡張
MAX_CAPACITY = st.sidebar.slider("エリア内の最大許容人数 (100%基準)", 5, 200, 150)

col_video, col_status = st.columns([2, 1])
with col_video:
    st.subheader("📹 モニタリング映像")
    frame_placeholder = st.empty()
with col_status:
    st.subheader("📊 リアルタイム分析")
    metrics_placeholder = st.empty()
    alert_placeholder = st.empty()


# ==============================================================================
# 6. メイン処理（2.5倍ズームAI・高安定化ライフサイクル）
# ==============================================================================
if run_system:
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        st.error(f"画像ファイル '{IMAGE_PATH}' が読み込めませんでした。")
        st.stop()

    h, w = frame.shape[:2]
    BASE_ROWS, BASE_COLS = 12, 16
    c_w = w / BASE_COLS
    c_h = h / BASE_ROWS

    # [STEP 1] エリア設定（初回のみ実行）
    if st.session_state.rough_grid is None:
        with st.spinner("エリア設定画面を表示しています..."):
            rough_grid = setup_fence_area_popup(frame, rows=BASE_ROWS, cols=BASE_COLS)
            st.session_state.rough_grid = rough_grid

            fence_mask = np.zeros((h, w), dtype=np.uint8)
            for r in range(BASE_ROWS):
                for c in range(BASE_COLS):
                    if rough_grid[r, c]:
                        pt1 = (int(c * c_w), int(r * c_h))
                        pt2 = (int((c + 1) * c_w), int((r + 1) * c_h))
                        cv2.rectangle(fence_mask, pt1, pt2, 255, -1)
            st.session_state.fence_mask = fence_mask
            st.rerun()

    rough_grid = st.session_state.rough_grid
    fence_mask = st.session_state.fence_mask
    inverse_mask = cv2.bitwise_not(fence_mask)

    # [STEP 2] 2.5倍グリッドズーム推論（重い処理なので初回のみ実行しキャッシュして固まるのを防止）
    if st.session_state.final_people_boxes is None:
        with st.spinner("AIが指定グリッド内を2.5倍ズームスキャン中（最初だけ数秒かかります）..."):
            processed_frame, brightness, is_night = preprocess_frame(frame, BRIGHTNESS_THRESHOLD)
            all_boxes = []
            
            for r in range(BASE_ROWS):
                for c in range(BASE_COLS):
                    if rough_grid[r, c]:
                        # 境界線の見落としを防ぐための重なりバッファ
                        buffer = 35
                        x_start = max(0, int(c * c_w) - buffer)
                        y_start = max(0, int(r * c_h) - buffer)
                        x_end = min(w, int((c + 1) * c_w) + buffer)
                        y_end = min(h, int((r + 1) * c_h) + buffer)
                        
                        grid_crop = processed_frame[y_start:y_end, x_start:x_end]
                        if grid_crop.size == 0:
                            continue
                            
                        # 2.5倍ズーム処理
                        grid_zoomed = cv2.resize(grid_crop, (int(grid_crop.shape[1] * 2.5), int(grid_crop.shape[0] * 2.5)), interpolation=cv2.INTER_CUBIC)
                        
                        # しきい値を0.04まで下げて極小サイズを強引にハント
                        results = model(grid_zoomed, classes=[0], conf=0.04, verbose=False)
                        
                        if len(results) > 0 and len(results[0].boxes) > 0:
                            for box in results[0].boxes:
                                xyxy = box.xyxy[0].cpu().numpy()
                                # ズーム座標の全体逆引き復元
                                global_x1 = (xyxy[0] / 2.5) + x_start
                                global_y1 = (xyxy[1] / 2.5) + y_start
                                global_x2 = (xyxy[2] / 2.5) + x_start
                                global_y2 = (xyxy[3] / 2.5) + y_start
                                all_boxes.append([global_x1, global_y1, global_x2, global_y2, box.conf[0].cpu().numpy()])

            final_people_boxes = []
            if len(all_boxes) > 0:
                all_boxes = np.array(all_boxes)
                # NMSによる重複枠のクレンジング
                nms_indices = cv2.dnn.NMSBoxes(
                    all_boxes[:, :4].astype(int).tolist(), 
                    all_boxes[:, 4].astype(float).tolist(), 
                    score_threshold=0.04, 
                    nms_threshold=0.5
                )
                if len(nms_indices) > 0:
                    for i in nms_indices.flatten():
                        final_people_boxes.append(all_boxes[i])
                        
            st.session_state.final_people_boxes = final_people_boxes
            st.rerun()

    # [STEP 3] 高速レンダリング＆マイコン制御層
    final_people_boxes = st.session_state.final_people_boxes
    processed_frame, brightness, is_night = preprocess_frame(frame, BRIGHTNESS_THRESHOLD)

    output_image = frame.copy()
    people_in_wall = 0

    for box in final_people_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        center_x = int(x1 + ((x2 - x1) / 2))
        center_y = int(y1 + ((y2 - y1) / 2))

        check_x = min(max(0, center_x), w - 1)
        check_y = min(max(0, center_y), h - 1)

        # 指定エリア内に中心点があるかチェック
        if fence_mask[check_y, check_x] == 255:
            people_in_wall += 1
            color = (0, 0, 255) if is_night else (0, 255, 0)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 1)
            cv2.circle(output_image, (center_x, center_y), 2, color, -1)

    # 密度計算とマイコン送信
    density = (people_in_wall / MAX_CAPACITY) * 100
    is_triggered = density >= 90.0
    control_hardware_real(is_triggered, density)

    # エリア外の減光エフェクト
    dark_frame = (output_image * 0.3).astype(np.uint8)
    highlighted_area = cv2.bitwise_and(output_image, output_image, mask=fence_mask)
    darkened_area = cv2.bitwise_and(dark_frame, dark_frame, mask=inverse_mask)
    monitoring_display = cv2.add(highlighted_area, darkened_area)

    # UI更新
    with metrics_placeholder.container():
        st.metric(label="現在の壁内人数 (AI精密ズームカウント)", value=f"{people_in_wall} 人")
        st.metric(label="現在の人口密度", value=f"{density:.1f} %", delta=f"{density - 90.0:.1f} %" if is_triggered else None, delta_color="inverse")
        
    if is_triggered:
        alert_placeholder.error(f"🚨 **警告: 密度が90%以上です ({density:.1f}%)**\n\nマイコン（COM3）に作動信号 'H' を送信中！")
    else:
        alert_placeholder.success(f"✅ **安全: 正常な密度です ({density:.1f}%)**\n\nLEDは消灯（消灯命令 'L'）しています。")
        
    frame_rgb = cv2.cvtColor(monitoring_display, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    
    time.sleep(0.1)
    st.rerun()

else:
    # チェックを外したら全キャッシュをクリア
    st.session_state.rough_grid = None
    st.session_state.fence_mask = None
    st.session_state.final_people_boxes = None
    frame_placeholder.info("サイドバーの「システムを起動」にチェックを入れてください。")
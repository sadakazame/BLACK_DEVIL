import time
import cv2
import numpy as np
import serial  # シリアル通信用のライブラリ
import streamlit as st

# ページの設定
st.set_page_config(page_title="壁内人口密度監視システム", layout="wide")

# ==============================================================================
# 0. マイコン（シリアルポート）の接続設定
# ==============================================================================
# Windowsの場合、デバイスマネージャーで「STLink Virtual COM Port」のCOM番号を確認してください（例: 'COM7'）
SERIAL_PORT = "COM7"
BAUD_RATE = 38400  # huart2.Init.BaudRate = 38400

@st.cache_resource
def init_serial():
    """ポートが既に開いている場合はそれを再利用し、エラー時は一度閉じて再接続します"""
    try:
        # 既存のグローバル変数からポートが生きているか確認
        if 'ser' in globals() and ser is not None and ser.is_open:
            return ser
        
        # 新規接続を確立
        new_ser = serial.Serial()
        new_ser.port = SERIAL_PORT
        new_ser.baudrate = BAUD_RATE
        new_ser.timeout = 1
        new_ser.open()
        
        time.sleep(2)  # 接続安定のためのウェイト
        return new_ser
    except Exception as e:
        # エラー発生時のデバッグ用に、ターミナルへ本当のエラー内容を出力します
        print(f"\n[⚠️ SERIAL ERROR] {SERIAL_PORT} への接続失敗詳細: {e}\n")
        st.sidebar.warning(f"マイコン ({SERIAL_PORT}) が未接続です。実機なしモードで起動します。")
        return None

# シリアルポートの初期化
ser = init_serial()

# ==============================================================================
# 1. 設定
# ==============================================================================
IMAGE_PATH = "noonlot.png"

# ==============================================================================
# 2. 状態管理（セッション状態の初期化）
# ==============================================================================
if "last_hardware_state" not in st.session_state:
    st.session_state.last_hardware_state = False
if "fence_mask" not in st.session_state:
    st.session_state.fence_mask = None
if "rough_grid" not in st.session_state:
    st.session_state.rough_grid = None

# コールバック用の一時変数
grid_state_ui = None
cell_w_ui = 0
cell_h_ui = 0
drawing_ui = False

# ==============================================================================
# 3. エリア設定用の関数（OpenCVポップアップ）
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
        if key == 13 or key == 32:
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyWindow(window_name)
    return grid_state_ui


# ==============================================================================
# 4. あなたの担当：マイコン制御送信ロジック（統合・実機連動仕様）
# ==============================================================================
def control_hardware_real(activate: bool, current_density: float):
    """
    実際のマイコンへ信号を送信し、同時にターミナルにもログを残す関数
    """
    # 混雑度がしきい値以上かつ、前回「停止」状態だった場合（ONになった瞬間）
    if activate and not st.session_state.last_hardware_state:
        print(f"\n[🔴 HARDWARE] 警告レベル到達 ({current_density:.1f}%) -> マイコンに 'H' 送信")
        if ser and ser.is_open:
            ser.write(b"H")  # マイコンへ点灯命令(High)を送信
        st.session_state.last_hardware_state = True

    # 混雑度がしきい値未満かつ、前回「作動」状態だった場合（OFFになった瞬間）
    elif not activate and st.session_state.last_hardware_state:
        print(f"\n[🟢 HARDWARE] 安全レベル復帰 ({current_density:.1f}%) -> マイコンに 'L' 送信")
        if ser and ser.is_open:
            ser.write(b"L")  # マイコンへ消灯命令(Low)を送信
        st.session_state.last_hardware_state = False


# ==============================================================================
# 5. Streamlit UI のレイアウト構築
# ==============================================================================
st.title("🛡️ 壁内人口密度監視システム (背景色参照・実機連動版)")
st.markdown("指定エリア内の「芝生の緑色」がどれだけ群衆で遮られているかをピクセル単位で計算し、マイコンのLEDを制御します。")

st.sidebar.header("⚙️ システム設定")
run_system = st.sidebar.checkbox("システムを起動（ループ開始）", value=False)

# 警告を発動させる混雑度（％）のしきい値を設定
ALERT_THRESHOLD = st.sidebar.slider("警告を出す混雑度のしきい値 (%)", 10, 100, 40)

col_video, col_status = st.columns([2, 1])
with col_video:
    st.subheader("📹 モニタリング映像")
    frame_placeholder = st.empty()
with col_status:
    st.subheader("📊 リアルタイム分析")
    metrics_placeholder = st.empty()
    alert_placeholder = st.empty()

# ==============================================================================
# 6. メイン処理（背景参照型・超高速演算）
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

    # エリア設定（初回のみ）
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

    # 背景参照（色領域）演算
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    total_area_pixels = cv2.countNonZero(fence_mask)

    if total_area_pixels > 0:
        green_in_area = cv2.bitwise_and(green_mask, green_mask, mask=fence_mask)
        green_pixels = cv2.countNonZero(green_in_area)
        green_ratio = green_pixels / total_area_pixels
        density = (1.0 - green_ratio) * 100
    else:
        density = 0.0

    # 【重要】判定フラグの作成と実際のマイコン制御関数の呼び出し
    is_triggered = density >= ALERT_THRESHOLD
    control_hardware_real(is_triggered, density)

    # 視覚化エフェクト（非緑色部分を赤く強調）
    output_image = frame.copy()
    non_green_in_area = cv2.bitwise_and(cv2.bitwise_not(green_mask), cv2.bitwise_not(green_mask), mask=fence_mask)
    output_image[non_green_in_area == 255] = output_image[non_green_in_area == 255] * 0.7 + np.array([0, 0, 70])

    # マスクブレンド（エリア外を薄暗く）
    dark_frame = (output_image * 0.3).astype(np.uint8)
    highlighted_area = cv2.bitwise_and(output_image, output_image, mask=fence_mask)
    darkened_area = cv2.bitwise_and(dark_frame, dark_frame, mask=inverse_mask)
    monitoring_display = cv2.add(highlighted_area, darkened_area)

    # UI更新
    with metrics_placeholder.container():
        st.metric(label="指定エリア内の総面積", value=f"{total_area_pixels:,} px")
        st.metric(label="現在の群衆混雑度 (面積比率)", value=f"{density:.1f} %")
        
    if is_triggered:
        alert_placeholder.error(f"🚨 **警告: 混雑度が設定値以上です ({density:.1f}%)**\n\nマイコンに点灯命令 'H' を送信しました。")
    else:
        alert_placeholder.success(f"✅ **安全: 快適な状態です ({density:.1f}%)**\n\nLEDは消灯（消灯命令 'L'）しています。")
        
    frame_rgb = cv2.cvtColor(monitoring_display, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    
    time.sleep(0.1)
    st.rerun()
else:
    st.session_state.rough_grid = None
    st.session_state.fence_mask = None
    frame_placeholder.info("サイドバーの「システムを起動」にチェックを入れてください。")
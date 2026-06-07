import time
import cv2
import numpy as np
import streamlit as st

# ページの設定
st.set_page_config(page_title="壁内人口密度監視システム", layout="wide")

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
        if key == 13 or key == 32: break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break
    cv2.destroyWindow(window_name)
    return grid_state_ui

# ==============================================================================
# 4. あなたの担当：ターミナル模擬出力
# ==============================================================================
def control_hardware_simulation(activate: bool, current_density: float):
    if activate and not st.session_state.last_hardware_state:
        print(f"\n[🔴 HARDWARE ACTIVATE] 警告: 面積密度が {current_density:.1f}% に達しました！")
        st.session_state.last_hardware_state = True
    elif not activate and st.session_state.last_hardware_state:
        print(f"\n[🟢 HARDWARE STOP] 安全: 面積密度が {current_density:.1f}% に低下しました。")
        st.session_state.last_hardware_state = False

# ==============================================================================
# 5. Streamlit UI のレイアウト構築
# ==============================================================================
st.title("🛡️ 壁内人口密度監視システム (背景色参照・面積比率版)")
st.markdown("AIを使わず、指定エリア内の「芝生の緑色」がどれだけ群衆で遮られているかをピクセル単位でリアルタイム計算します。")

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

    # --- ★ ここが背景参照（色領域）ロジックの核心 ---
    # 1. 画像をHSV（色を扱いやすい空間）に変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. 今回の画像の「芝生の緑色」の範囲を定義 (H:色相, S:彩度, V:明度)
    # ※画像の芝生の色合いに合わせて調整しています
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([85, 255, 255])

    # 3. 画像全体から「緑色」の部分だけを白(255)にしたマスクを作る
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # 4. ユーザーが指定したエリア（fence_mask）内の、全体のピクセル数を数える
    total_area_pixels = cv2.countNonZero(fence_mask)

    if total_area_pixels > 0:
        # 5. 指定エリア内に残っている「緑色のピクセル数」を計算
        green_in_area = cv2.bitwise_and(green_mask, green_mask, mask=fence_mask)
        green_pixels = cv2.countNonZero(green_in_area)

        # 6. 混雑度（＝緑色じゃない部分の比率）を割り出す
        green_ratio = green_pixels / total_area_pixels
        density = (1.0 - green_ratio) * 100  # 人が詰まるほど100%に近づく
    else:
        density = 0.0

    # 判定とアラート
    is_triggered = density >= ALERT_THRESHOLD
    control_hardware_simulation(is_triggered, density)

    # モニタリング表示用に、検知された「人（非緑色）」の部分に薄く赤ノイズを乗せる可視化
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
        st.metric(label="指定エリア内の面積", value=f"{total_area_pixels:,} px")
        st.metric(label="現在の群衆混雑度 (面積比率)", value=f"{density:.1f} %")
        
    if is_triggered:
        alert_placeholder.error(f"🚨 **警告: 混雑度が設定値以上です ({density:.1f}%)**\n\nマイコン模擬信号 'H' を送信中。")
    else:
        alert_placeholder.success(f"✅ **安全: 快適な状態です ({density:.1f}%)**\n\nシステム監視中...")
        
    frame_rgb = cv2.cvtColor(monitoring_display, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    
    time.sleep(0.1)
    st.rerun()
else:
    st.session_state.rough_grid = None
    st.session_state.fence_mask = None
    frame_placeholder.info("サイドバーの「システムを起動」にチェックを入れてください。")
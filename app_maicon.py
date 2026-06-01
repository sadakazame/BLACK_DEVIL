import cv2
import numpy as np
import streamlit as st
import time
import serial  # シリアル通信用のライブラリを追加

# ページの設定
st.set_page_config(page_title="壁内人口密度監視システム", layout="wide")

# ==========================================
# 0. マイコン（シリアルポート）の接続設定
# ==========================================
# Windowsの場合、デバイスマネージャーで「STLink Virtual COM Port」のCOM番号を確認してください（例: 'COM3'）
# MacやLinuxの場合は '/dev/tty.usbmodem...' や '/dev/ttyACM0' などになります
SERIAL_PORT = 'COM7' 
BAUD_RATE = 38400  # 資料20ページの huart2.Init.BaudRate = 38400 に合わせています

@st.cache_resource
def init_serial():
    """Streamlitのリロードでも接続を切断しないよう、ポートをキャッシュ化します"""
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # 接続安定のためのウェイト
        return ser
    except Exception as e:
        st.error(f"マイコン（{SERIAL_PORT}）への接続に失敗しました。ポート番号を確認してください。: {e}")
        return None

# シリアルポートの初期化
ser = init_serial()

# ==========================================
# 1. 状態管理（セッション状態の初期化）
# ==========================================
if 'last_hardware_state' not in st.session_state:
    st.session_state.last_hardware_state = False

# ==========================================
# 2. 他の人からロジックをもらう想定の関数
# ==========================================
def detect_walls(frame):
    h, w, _ = frame.shape
    wall_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(wall_mask, (int(w*0.2), int(h*0.2)), (int(w*0.8), int(h*0.8)), 255, -1)
    return wall_mask

def detect_people(frame, demo_people_count=0):
    people_boxes = []
    preset_positions = [
        [200, 150, 240, 230], [300, 160, 340, 240], [400, 150, 440, 230],
        [220, 250, 260, 330], [320, 260, 360, 340], [420, 250, 460, 330],
        [150, 200, 190, 280], [450, 200, 490, 280], [250, 200, 290, 280],
        [350, 200, 390, 280]
    ]
    for i in range(min(demo_people_count, len(preset_positions))):
        people_boxes.append(preset_positions[i])
    return people_boxes

# ==========================================
# 3. あなたの担当：密度計算 ＆ 判定ロジック
# ==========================================
def calculate_density_and_judge(wall_mask, people_boxes):
    people_in_wall = 0
    for box in people_boxes:
        x1, y1, x2, y2 = box
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if wall_mask[cy, cx] == 255:
            people_in_wall += 1

    MAX_CAPACITY = 10 
    density = (people_in_wall / MAX_CAPACITY) * 100
    is_triggered = density >= 90.0
    return density, is_triggered, people_in_wall

# ==========================================
# 4. あなたの担当：マイコン制御送信ロジック（追加・更新）
# ==========================================
def control_hardware_real(activate: bool, current_density: float):
    """
    実際のマイコンへ信号を送信し、同時にターミナルにもログを残す関数
    """
    # 密度90%以上かつ、前回「停止」状態だった場合（ONになった瞬間）
    if activate and not st.session_state.last_hardware_state:
        print(f"\n[🔴 HARDWARE] 警告レベル到達 ({current_density:.1f}%) -> マイコンに 'H' 送信")
        if ser and ser.is_open:
            ser.write(b'H')  # マイコンへ点灯命令を送信
        st.session_state.last_hardware_state = True
        
    # 密度90%未満かつ、前回「作動」状態だった場合（OFFになった瞬間）
    elif not activate and st.session_state.last_hardware_state:
        print(f"\n[🟢 HARDWARE] 安全レベル復帰 ({current_density:.1f}%) -> マイコンに 'L' 送信")
        if ser and ser.is_open:
            ser.write(b'L')  # マイコンへ消灯命令を送信
        st.session_state.last_hardware_state = False

# ==========================================
# 5. Streamlit UI のレイアウト構築
# ==========================================
st.title("🛡️ 壁内人口密度監視システム (実機連動版)")

st.sidebar.header("⚙️ システム設定")
run_system = st.sidebar.checkbox("システムを起動（ループ開始）", value=False)

st.sidebar.subheader("🧪 デモ用テスト設定")
demo_count = st.sidebar.slider("シミュレートする人数（9人以上でLED点灯）", 0, 10, 5)

col_video, col_status = st.columns([2, 1])
with col_video:
    st.subheader("📹 モニタリング映像")
    frame_placeholder = st.empty()
with col_status:
    st.subheader("📊 リアルタイム分析")
    metrics_placeholder = st.empty()
    alert_placeholder = st.empty()

# ==========================================
# 6. メイン実行ループ
# ==========================================
if run_system:
    while True:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 各種ロジック処理
        wall_mask = detect_walls(frame)
        people_boxes = detect_people(frame, demo_count)
        density, is_triggered, count = calculate_density_and_judge(wall_mask, people_boxes)
        
        # 【重要】実際のマイコン制御関数を呼び出し
        control_hardware_real(is_triggered, density)
        
        # 描画
        frame[wall_mask == 255] = frame[wall_mask == 255] * 0.5 + np.array([60, 20, 20]) 
        for box in people_boxes:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
            
        # UI書き換え
        with metrics_placeholder.container():
            st.metric(label="現在の壁内人数", value=f"{count} 人")
            st.metric(label="現在の人口密度", value=f"{density:.1f} %")
            
        if is_triggered:
            alert_placeholder.error("🚨 **警告: 密度が90%以上です！**\n\nマイコンのLEDを点灯させています。")
        else:
            alert_placeholder.success("✅ **安全: 正常な密度です**\n\nLEDは消灯しています。")
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        time.sleep(0.1)
else:
    frame_placeholder.info("サイドバーの「システムを起動」にチェックを入れてください。")
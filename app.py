import cv2
import numpy as np
import streamlit as st
import time

# ページの設定
st.set_page_config(page_title="壁内人口密度監視システム", layout="wide")

# ==========================================
# 1. 状態管理（セッション状態の初期化）
# ==========================================
# ハードウェアの前回の状態を記録（ターミナルの連打防止用）
if 'last_hardware_state' not in st.session_state:
    st.session_state.last_hardware_state = False

# ==========================================
# 2. 他の人からロジックをもらう想定の関数（プレースホルダー）
# ==========================================
def detect_walls(frame):
    """
    【他者担当】壁を検出してマスク画像を返す関数（デモ用に中央に四角い壁を作ります）
    """
    h, w, _ = frame.shape
    wall_mask = np.zeros((h, w), dtype=np.uint8)
    # 画面中央付近に「壁に囲まれたエリア」を擬似的に作成
    cv2.rectangle(wall_mask, (int(w*0.2), int(h*0.2)), (int(w*0.8), int(h*0.8)), 255, -1)
    return wall_mask

def detect_people(frame, demo_people_count=0):
    """
    【他者担当】人を検出してバウンディングボックスのリストを返す関数
    （デモ用にスライダーの値に応じた人数の座標を擬似的に生成します）
    """
    people_boxes = []
    # デモ用に固定の座標リストから、指定された人数分だけ取り出す
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
    """
    壁の領域に対して、人がどれだけ密度を占めているかを計算・判定する関数
    """
    people_in_wall = 0
    
    for box in people_boxes:
        x1, y1, x2, y2 = box
        # 人の中心点を計算
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        # 中心の座標が壁のマスク内（値が255の場所）にあるか判定
        if wall_mask[cy, cx] == 255:
            people_in_wall += 1

    # 【密度の定義】ここでは仮に「最大収容人数 10人」に対する割合として計算
    MAX_CAPACITY = 10 
    density = (people_in_wall / MAX_CAPACITY) * 100
    
    # 密度が 90% 以上かどうかの判定フラグ
    is_triggered = density >= 90.0
    
    return density, is_triggered, people_in_wall

# ==========================================
# 4. あなたの担当：ターミナル模擬出力（実機なし用）
# ==========================================
def control_hardware_simulation(activate: bool, current_density: float):
    """
    実機がない環境用：状態が変化した瞬間だけターミナル（コンソール）にログを出す
    """
    if activate and not st.session_state.last_hardware_state:
        print("\n" + "="*50)
        print(f"[🔴 HARDWARE ACTIVATE] 警告: 密度が {current_density:.1f}% に達しました！")
        print(">> LED点灯 / モータ駆動 信号を送信しました。")
        print("="*50 + "\n")
        st.session_state.last_hardware_state = True # 状態を「作動中」に更新
        
    elif not activate and st.session_state.last_hardware_state:
        print("\n" + "="*50)
        print(f"[🟢 HARDWARE STOP] 安全: 密度が {current_density:.1f}% に低下しました。")
        print(">> デバイスを停止しました。")
        print("="*50 + "\n")
        st.session_state.last_hardware_state = False # 状態を「停止中」に更新

# ==========================================
# 5. Streamlit UI のレイアウト構築
# ==========================================
st.title("🛡️ 壁内人口密度監視システム")
st.markdown("他のメンバーの検出コードを統合し、指定エリア内の密度をリアルタイムに監視・判定します。")

# サイドバーに設定とテスト用のコントロールを配置
st.sidebar.header("⚙️ システム設定")
run_system = st.sidebar.checkbox("システムを起動（ループ開始）", value=False)

st.sidebar.subheader("🧪 デモ用テスト設定")
# 他の人のコードが動いている状態を再現するための人数調整スライダー（最大10人）
demo_count = st.sidebar.slider("シミュレートする人数（9人以上で90%到達）", 0, 10, 5)

# メイン画面のレイアウト（左に映像、右にステータス）
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
        # ベースとなる真っ黒な画像を生成（カメラ映像の代わり）
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # [STEP 1] 他者の検出ロジック呼び出し
        wall_mask = detect_walls(frame)
        people_boxes = detect_people(frame, demo_count) # スライダーの値を渡してダミー生成
        
        # [STEP 2] 自身の判定ロジック呼び出し
        density, is_triggered, count = calculate_density_and_judge(wall_mask, people_boxes)
        
        # [STEP 3] ハードウェア制御（ターミナル出力）
        control_hardware_simulation(is_triggered, density)
        
        # [STEP 4] 映像への描画処理（可視化）
        # 壁のエリアを薄い青色で半透明に塗る
        frame[wall_mask == 255] = frame[wall_mask == 255] * 0.5 + np.array([60, 20, 20]) 
        
        # 人のバウンディングボックスを赤色の枠で描く
        for box in people_boxes:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            # 中心の点を描画
            cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
            
        # [STEP 5] Streamlit UIの書き換え
        # メトリクス（数値）の更新
        with metrics_placeholder.container():
            st.metric(label="現在の壁内人数", value=f"{count} 人")
            st.metric(label="現在の人口密度", value=f"{density:.1f} %", delta=f"{density - 90.0:.1f} %" if is_triggered else None, delta_color="inverse")
            
        # 警告アラートの更新
        if is_triggered:
            alert_placeholder.error("🚨 **警告: 密度が90%以上です！**\n\nターミナルに作動信号を出力しました。")
        else:
            alert_placeholder.success("✅ **安全: 正常な密度です**\n\nシステム監視中...")
            
        # 映像をBGRからRGBに変換してStreamlitに表示
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # ループの間隔（負荷軽減のために約0.1秒待つ）
        time.sleep(0.1)
else:
    frame_placeholder.info("サイドバーの「システムを起動」にチェックを入れてください。")
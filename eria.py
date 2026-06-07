# カメラ映像から入力
import cv2
import numpy as np

# --- グローバル変数（マウス操作用） ---
grid_state = None
cell_w = 0
cell_h = 0
drawing = False


def select_grid_callback(event, x, y, flags, param):

    #マウス操作を受け取り、グリッドのON/OFFを切り替える関数

    global grid_state, cell_w, cell_h, drawing

    # グリッドのどの行・列をクリックしているか計算
    col = x // cell_w
    row = y // cell_h

    # 画面外をクリックした際のエラー防止
    if row >= grid_state.shape[0] or col >= grid_state.shape[1]:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        # クリックしたセルの状態を反転（ONならOFF、OFFならON）
        grid_state[row, col] = not grid_state[row, col]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # ドラッグ中はセルをONに塗りつぶす（操作性向上のため）
            grid_state[row, col] = True

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


def setup_fence_area(frame, rows=10, cols=15):

    #ユーザーにCAPTCHA風の画面を提示し、マスク画像を作成する関数

    global grid_state, cell_w, cell_h

    h, w = frame.shape[:2]
    cell_w = w // cols
    cell_h = h // rows

    # 全てのグリッド状態をFalse（未選択）で初期化
    grid_state = np.zeros((rows, cols), dtype=bool)

    window_name = "Select Fence Area (Press ENTER to confirm)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, select_grid_callback)

    print("\n" + "=" * 40)
    print("【エリア設定モード】")
    print("マウスでクリックまたはドラッグして、観客エリアを選択してください。")
    print("選択が完了したら 'Enter' キーを押してください。")
    print("=" * 40 + "\n")

    while True:
        display = frame.copy()
        overlay = frame.copy()

        # 選択されたセルを赤色で塗りつぶす
        for r in range(rows):
            for c in range(cols):
                if grid_state[r, c]:
                    pt1 = (c * cell_w, r * cell_h)
                    pt2 = ((c + 1) * cell_w, (r + 1) * cell_h)
                    cv2.rectangle(overlay, pt1, pt2, (0, 0, 255), -1)

        # 半透明にして元の映像と合成（アルファブレンド）
        cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)

        # グリッドの線を描画
        for r in range(1, rows):
            cv2.line(display, (0, r * cell_h), (w, r * cell_h), (255, 255, 255), 1)
        for c in range(1, cols):
            cv2.line(display, (c * cell_w, 0), (c * cell_w, h), (255, 255, 255), 1)

        cv2.imshow(window_name, display)

        # Enterキー(13) または Spaceキー(32) で決定
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 32:
            break

    cv2.destroyWindow(window_name)

    # --- 選択結果から最終的なマスク画像（白黒）を生成 ---
    final_mask = np.zeros((h, w), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if grid_state[r, c]:
                pt1 = (c * cell_w, r * cell_h)
                pt2 = ((c + 1) * cell_w, (r + 1) * cell_h)
                cv2.rectangle(final_mask, pt1, pt2, 255, -1)

    return final_mask


# ==========================================
# テスト実行部分（カメラ入力版）
# ==========================================
if __name__ == "__main__":
    print("カメラを起動しています...")
    cap = cv2.VideoCapture(0)  # 必要に応じて 1 や 2 に変更

    if not cap.isOpened():
        print("【エラー】カメラにアクセスできませんでした。")
        exit()

    # --- [STEP 1] カメラの映像が安定するまで数フレーム読み飛ばす ---
    for _ in range(5):
        cap.read()

    ret, initial_frame = cap.read()
    if not ret:
        print("フレームの取得に失敗しました。")
        exit()

    # --- [STEP 2] 初期セットアップ：手動でマスクを作成（1回だけ実行） ---
    # rows(行数) と cols(列数) を変更するとグリッドの細かさを調整できます
    fence_mask = setup_fence_area(initial_frame, rows=12, cols=16)

    print("エリアの設定が完了しました。監視をスタートします...")
    print("「q」キーを押すと終了します。")

    # --- [STEP 3] メインループ：以降はこのマスクを使い回す ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ここで、取得した fence_mask を使って「領域内の人の数」などを判定します。
        # 例として、設定したエリア内だけ明るく表示し、エリア外を暗くする処理を記述します。

        # エリア外を暗くする視覚効果（ビット演算）
        dark_frame = (frame * 0.3).astype(np.uint8)
        # maskが255(白)の部分だけ元の明るいframeを残す
        highlighted_area = cv2.bitwise_and(frame, frame, mask=fence_mask)
        # maskが0(黒)の部分は暗いframeを残す
        inverse_mask = cv2.bitwise_not(fence_mask)
        darkened_area = cv2.bitwise_and(dark_frame, dark_frame, mask=inverse_mask)

        # 結合して表示
        monitoring_display = cv2.add(highlighted_area, darkened_area)

        # 結果の可視化
        cv2.imshow("Monitoring (Area highlighted)", monitoring_display)
        # cv2.imshow("Static Mask", fence_mask) # マスク単体を見たい場合はコメントアウトを外す

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("テストを終了しました。")
import osmnx as ox
import pandas as pd
from pybdshadow import bdshadow_sunlight
import matplotlib.pyplot as plt

def simulate_real_map_shadows(lat, lon, dist, target_datetime):
    print(f"\n[1/4] 地図データを取得中... ({lat}, {lon})")
    try:
        # 建物データの取得
        buildings = ox.features_from_point((lat, lon), dist=dist, tags={'building': True})
        # 道路ネットワークの取得
        roads = ox.graph_from_point((lat, lon), dist=dist, network_type='all')
        # 水路・鉄道などの取得
        leisure = ox.features_from_point((lat, lon), dist=dist, tags={'water': True, 'railway': True, 'leisure': 'park'})
    except Exception as e:
        print(f"データの取得に失敗しました: {e}")
        return

    # 1. 建物の整形
    buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
    buildings = buildings.explode(index_parts=True)
    buildings['building_id'] = range(len(buildings))

    # 2. 高さの設定
    if 'height' not in buildings.columns:
        buildings['height'] = 15.0
    buildings['height'] = buildings['height'].fillna(15.0).apply(
        lambda x: float(str(x).split()[0].replace('m', '')) if x else 15.0
    )

    # 3. 日時を設定
    date = pd.to_datetime(target_datetime).tz_localize('Asia/Tokyo')

    print(f"[2/4] {date.strftime('%H:%M')} の影を計算中...")
    try:
        shadows = bdshadow_sunlight(buildings, date)

        # 4. 可視化
        print("[3/4] 地図をレンダリングしています...")
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_facecolor('#f0f0f0') # 背景を薄いグレーに

        # 水路・公園などを描画
        if not leisure.empty:
            leisure.plot(ax=ax, color='#add8e6', alpha=0.5) # 水色は青系

        # 道路を描画（線として描画）
        ox.plot_graph(roads, ax=ax, node_size=0, edge_color='#ffffff', edge_linewidth=1.5, show=False, close=False)

        # 建物を描画
        buildings.plot(ax=ax, color='#888888', edgecolor='#666666', linewidth=0.5, alpha=0.8)
        
        # 影を描画
        if not shadows.empty:
            shadows.plot(ax=ax, color='#222222', alpha=0.5, zorder=10)
        
        plt.title(f"Detailed Urban Shadow Analysis: {target_datetime}")
        print("[4/4] 完了しました。")
        plt.show()
        
    except Exception as e:
        print(f"計算中にエラーが発生しました: {e}")

# --- メイン処理 ---
if __name__ == "__main__":
    LAT, LON = 34.7024, 135.4959  # 大阪駅
    BASE_DATE = '2026-04-27'
    
    print("--- 都市インフラ連動・影シミュレーター ---")
    user_time = input("時間を入力してください (例 16:00): ")
    full_datetime = f"{BASE_DATE} {user_time}:00"
    
    simulate_real_map_shadows(LAT, LON, dist=400, target_datetime=full_datetime)
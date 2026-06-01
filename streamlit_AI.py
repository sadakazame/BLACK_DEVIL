import streamlit as st
import geopandas as gpd
import pandas as pd
import pydeck as pdk
import xgboost as xgb
import numpy as np
import pvlib
from shapely.affinity import translate
from shapely.geometry import MultiPoint

# --- 1. 基本設定 ---
LAT, LON = 35.694, 139.753


@st.cache_resource
def load_all_assets():
    # モデル読み込み
    model = xgb.XGBRegressor()
    model.load_model("urban_temp_model.json")

    # 道路データの読み込みとフィルタリング
    roads = gpd.read_file("chiyoda_road_with_satellite_temp.geojson").to_crs(epsg=4326)
    # 千代田区の範囲外（新宿区など）をカット
    roads = roads.cx[139.730:139.775, 35.670:35.705].copy()

    roads['centroid'] = roads.geometry.centroid
    roads['lat'] = roads['centroid'].y
    roads['lon'] = roads['centroid'].x

    # 建物データの読み込み
    bldgs = gpd.read_file("plateau_polygons.geojson").to_crs(epsg=4326)

    # 【重要】データのクレンジング
    # -9999.0 などのマイナス値や 0m を、標準的なビルの高さ(10m)に置き換え
    bldgs.loc[bldgs['height'] <= 0, 'height'] = 10.0
    # 異常に高い値もリミッターをかける
    bldgs.loc[bldgs['height'] > 300, 'height'] = 10.0

    # 軽量化
    bounds = roads.total_bounds
    bldgs = bldgs.cx[bounds[0] - 0.001:bounds[2] + 0.001, bounds[1] - 0.001:bounds[3] + 0.001].copy()
    bldgs = bldgs[['height', 'geometry']].copy()

    return model, roads, bldgs


model, roads, bldgs = load_all_assets()


# --- 2. 影計算ロジック（スケール修正版） ---
def get_shadows_and_judgement(date, hour, bldgs_df, roads_gdf):
    time_str = f"{date} {hour:02d}:00:00"
    time_obj = pd.to_datetime([time_str]).tz_localize('Asia/Tokyo')
    site = pvlib.location.Location(LAT, LON, tz='Asia/Tokyo')
    sol = site.get_solarposition(time_obj)
    alt, az = sol['elevation'].values[0], sol['azimuth'].values[0]

    # 太陽が低い時の暴走防止
    calc_alt = max(alt, 10.0)
    if alt <= 0: return None, alt, az, np.zeros(len(roads_gdf))

    rad = np.radians(az + 180)
    shadow_len_factor = 1.0 / np.tan(np.radians(calc_alt))

    # 1mあたりの緯度経度変換係数
    off_x_per_meter = 1.0 / 90600
    off_y_per_meter = 1.0 / 110900

    shadow_polys = []
    MAX_SHADOW_METERS = 150.0  # 影の最大長を150mに制限

    for _, row in bldgs_df.iterrows():
        try:
            h = row['height']
            s_len = min(h * shadow_len_factor, MAX_SHADOW_METERS)

            shift_x = np.sin(rad) * s_len * off_x_per_meter
            shift_y = np.cos(rad) * s_len * off_y_per_meter

            poly = row.geometry
            shifted = translate(poly, xoff=shift_x, yoff=shift_y)

            combined_pts = list(poly.exterior.coords) + list(shifted.exterior.coords)
            shadow_polys.append(MultiPoint(combined_pts).convex_hull)
        except:
            continue

    if not shadow_polys:
        return None, alt, az, np.zeros(len(roads_gdf))

    shadow_union = gpd.GeoDataFrame(geometry=shadow_polys, crs="4326").geometry.unary_union
    s_flags = roads_gdf.geometry.intersects(shadow_union).astype(int).values

    return shadow_union, alt, az, s_flags


# --- 3. UIと描画 ---
st.set_page_config(layout="wide", page_title="千代田区路面温度シミュレーター")

with st.sidebar:
    st.header("シミュレーション設定")
    date = st.date_input("日付", pd.to_datetime("2025-08-01"))
    hour = st.slider("時刻", 6, 18, 12)
    road_w = st.slider("道路の太さ", 5, 40, 15)

shadow_geom, altitude, azimuth, s_flags = get_shadows_and_judgement(date, hour, bldgs, roads)

# AI推論
input_df = pd.DataFrame({
    'lat': roads['lat'], 'lon': roads['lon'],
    'solar_elevation': altitude, 'shadow_flag': s_flags
})
roads['temp_pred'] = np.clip(model.predict(input_df), 20, 60).round(1)
roads['shadow_flag'] = s_flags


# 高コントラストな色分け
def apply_color(row):
    temp = row['temp_pred']
    if row['shadow_flag'] == 1:
        t = np.clip((temp - 30) / 10, 0, 1)
        return [0, int(200 * t), 255, 255]  # 日陰：青〜水色
    else:
        t = np.clip((temp - 35) / 15, 0, 1)
        return [255, int(255 * (1 - t)), 0, 255]  # 日向：黄〜赤


roads['color'] = roads.apply(apply_color, axis=1)

# 地図表示
st.title(f" 路面温度予測: {hour}時 (太陽高度 {altitude:.1f}°)")

st.pydeck_chart(pdk.Deck(
    initial_view_state=pdk.ViewState(latitude=LAT, longitude=LON, zoom=16, pitch=45, bearing=azimuth - 180),
    layers=[
        pdk.Layer("GeoJsonLayer", bldgs, extruded=True, get_elevation="height", get_fill_color=[180, 180, 180, 150]),
        pdk.Layer("GeoJsonLayer", roads, get_line_color="color", get_line_width=road_w, pickable=True),
        pdk.Layer("GeoJsonLayer", gpd.GeoDataFrame(geometry=[shadow_geom], crs="4326") if shadow_geom else None,
                  get_fill_color=[0, 0, 0, 80], stroked=False)
    ],
    tooltip={"text": "予測温度: {temp_pred}℃\n日陰フラグ: {shadow_flag}"}
))
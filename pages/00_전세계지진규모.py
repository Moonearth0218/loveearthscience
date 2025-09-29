# pages/00_World_Quakes.py
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
from datetime import datetime

st.set_page_config(page_title="00 | World Quakes", layout="wide")
st.title("00. 🌐 전세계 지진 지도 (규모별 색) + 국가별 최신 3건")

# --- 사이드바 ---
with st.sidebar:
    st.header("📄 CSV 업로드")
    f = st.file_uploader("전세계 지진 CSV (latitude, longitude, magnitude, time)", type=["csv"])
    st.caption("time은 ISO8601 권장 (예: 2025-09-29T12:34:00Z)")

    st.header("🎯 표시 옵션")
    mag_min = st.slider("표시할 최소 규모", 0.0, 10.0, 2.5, 0.1)
    size_scale = st.slider("마커 크기 스케일", 1000, 20000, 7000, 500)

    st.header("🌍 국가 선택")
    st.caption("국가를 선택하면 해당 국가의 최신 지진 3건을 표로 보여줍니다.")

# --- 데이터 로딩 ---
if f is None:
    st.info("샘플 CSV를 Home 페이지에서 내려받아 업로드하세요.")
    st.stop()

try:
    df = pd.read_csv(f)
except Exception:
    f.seek(0)
    df = pd.read_csv(f, encoding="cp949")

# 컬럼 표준화 시도
def _find(colnames, candidates):
    low = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand in low: return low[cand]
    return None

lat = _find(df.columns, ["latitude","lat","y","위도"])
lon = _find(df.columns, ["longitude","lon","x","경도"])
mag = _find(df.columns, ["magnitude","mag","mw","규모"])
tim = _find(df.columns, ["time","origin_time","date","발생시각"])

missing = [c for c in [lat,lon,mag,tim] if c is None]
if missing:
    st.error("필수 컬럼(latitude/longitude/magnitude/time)을 찾을 수 없어요. 컬럼명을 확인하세요.")
    st.stop()

# 형변환
df[lat] = pd.to_numeric(df[lat], errors="coerce")
df[lon] = pd.to_numeric(df[lon], errors="coerce")
df[mag] = pd.to_numeric(df[mag], errors="coerce")
# time 파싱
def parse_time(x):
    try:
        return pd.to_datetime(x, utc=True, errors="coerce")
    except Exception:
        return pd.NaT
df[tim] = df[tim].apply(parse_time)

df = df.dropna(subset=[lat,lon,mag])
df = df[df[mag] >= mag_min].copy()

if len(df)==0:
    st.warning("조건을 만족하는 데이터가 없습니다.")
    st.stop()

# --- 국가 경계 GeoJSON 로딩 ---
geo_path = Path("data/world_countries_simplified.geojson")
if not geo_path.exists():
    st.error("data/world_countries_simplified.geojson 파일이 필요합니다. (간소화된 국가 경계 GeoJSON)")
    st.stop()

with open(geo_path, "r", encoding="utf-8") as fp:
    world_geo = json.load(fp)

# 국가 목록
countries = [f["properties"].get("ADMIN") or f["properties"].get("name") for f in world_geo["features"]]
countries = sorted(list({c for c in countries if c}))

# 선택 UI
sel_country = st.selectbox("국가 선택", options=["(선택하지 않음)"] + countries, index=0)

# --- 색상 맵 (규모별) ---
def color_for_mag(m):
    # 약한→강한 : 연한 노랑 → 주황 → 빨강
    if m < 4.0: return [255, 221, 87]
    if m < 5.0: return [255, 178, 70]
    if m < 6.0: return [255, 128, 0]
    if m < 7.0: return [255, 80, 0]
    return [220, 20, 60]  # 강한 빨강

df["_color"] = df[mag].apply(color_for_mag)
df["_size"] = np.clip((df[mag].fillna(0)+1.0) * size_scale, 1500, 25000)

# 중심뷰
center_lat = float(df[lat].mean())
center_lon = float(df[lon].mean())

# GeoJson 레이어(선택 국가 하이라이트)
geo_layers = []
if sel_country != "(선택하지 않음)":
    filt_features = [ft for ft in world_geo["features"]
                     if (ft["properties"].get("ADMIN")==sel_country or ft["properties"].get("name")==sel_country)]
    highlight_geo = {"type":"FeatureCollection","features":filt_features}
    geo_layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            data=highlight_geo,
            stroked=True,
            filled=True,
            get_fill_color=[0, 150, 255, 50],
            get_line_color=[0, 120, 255],
            line_width_min_pixels=1.5,
            pickable=False
        )
    )

# 포인트 레이어
points = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position=[lon, lat],
    get_radius="_size",
    get_fill_color="_color",
    radius_min_pixels=2,
    radius_max_pixels=40,
    pickable=True
)

tooltip = {
    "html": """
    <div style="font-size:14px">
      <b>규모:</b> {""" + mag + """}<br/>
      <b>위도:</b> {""" + lat + """}, <b>경도:</b> {""" + lon + """}<br/>
      <b>시간:</b> {""" + tim + """}
    </div>
    """,
    "style": {"backgroundColor":"white","color":"black"}
}

deck = pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=1.7),
    layers=geo_layers + [points],
    tooltip=tooltip
)
st.pydeck_chart(deck, use_container_width=True)

# --- 선택 국가의 최신 3건 (폴리곤 내부 판정) ---
st.subheader("🧭 선택 국가 최신 지진 3건")

def _point_in_ring(point, ring):
    # ray casting
    x, y = point
    inside = False
    n = len(ring)
    for i in range(n):
        x1,y1 = ring[i]
        x2,y2 = ring[(i+1)%n]
        cond = ((y1>y) != (y2>y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-12) + x1)
        if cond: inside = not inside
    return inside

def _point_in_poly(point, poly):
    # poly: [outer, hole1, hole2 ...]; 좌표는 [lon,lat]
    outer = [(p[0],p[1]) for p in poly[0]]
    if not _point_in_ring(point, outer): return False
    # 구멍 제외
    for hole in poly[1:]:
        if _point_in_ring(point, [(p[0],p[1]) for p in hole]): return False
    return True

def _in_multipolygon(point, multipoly):
    # multipoly: [[[ring1],[ring2],...], [[ring1],...], ...]
    for poly in multipoly:
        if _point_in_poly(point, poly): return True
    return False

def _filter_by_country(df_in, country_name):
    if country_name == "(선택하지 않음)": 
        return df_in.sort_values(by=tim, ascending=False).head(3)
    feats = [ft for ft in world_geo["features"]
             if (ft["properties"].get("ADMIN")==country_name or ft["properties"].get("name")==country_name)]
    if not feats:
        return pd.DataFrame(columns=df_in.columns)
    parts = []
    for _, row in d_


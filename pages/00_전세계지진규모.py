# pages/00_전세계지진규모.py
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="00 | 전세계 지진 규모 지도", layout="wide")
st.title("00. 🌐 전세계 지진 (규모별 색) + 국가별 최신 3건")

# --- 사이드바 ---
with st.sidebar:
    st.header("📄 데이터 선택")
    use_builtin = st.checkbox("data/국외지진목록*.xls 사용", value=True)
    f = None
    if not use_builtin:
        f = st.file_uploader("전세계 지진 CSV/XLS 업로드", type=["csv","xls","xlsx"])
    st.caption("※ 기본값: 레포의 data 폴더에 있는 '국외지진목록_*.xls' 사용")

    st.header("🎯 표시 옵션")
    mag_min = st.slider("표시 최소 규모", 0.0, 10.0, 2.5, 0.1)
    size_scale = st.slider("마커 크기 스케일", 1000, 20000, 7000, 500)

    st.header("🌍 국가 선택")
    sel_country = st.text_input("국가명 입력(예: Japan, Korea, United States 등)", "")

# --- 데이터 로딩 ---
def load_built_in_xls() -> pd.DataFrame:
    data_dir = Path("data")
    # 이름 패턴에 맞는 첫 파일 선택
    cand = list(data_dir.glob("국외지진목록*.xls"))
    if not cand:
        st.error("data/ 폴더에 '국외지진목록*.xls' 파일이 없습니다.")
        st.stop()
    path = cand[0]
    try:
        # .xls → xlrd 필요
        df = pd.read_excel(path)                # streamlit cloud에서 xlrd 설치 필요
    except Exception:
        df = pd.read_excel(path, engine="xlrd") # 강제 엔진
    return df

def load_user_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            return pd.read_csv(uploaded, encoding="cp949")
    else:  # xls/xlsx
        try:
            return pd.read_excel(uploaded)
        except Exception:
            uploaded.seek(0)
            return pd.read_excel(uploaded, engine="xlrd")

if use_builtin:
    df = load_built_in_xls()
else:
    if f is None:
        st.info("CSV/XLS 업로드 또는 상단 체크박스로 내장 파일을 사용하세요.")
        st.stop()
    df = load_user_file(f)

# --- 컬럼 매핑 (한국어/영문 유연 인식) ---
def find_col(cols, candidates):
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    # 부분일치(예: '발생시각(UTC)')
    for c in cols:
        lc = c.lower()
        if any(cand in lc for cand in candidates):
            return c
    return None

lat = find_col(df.columns, ["latitude","lat","위도"])
lon = find_col(df.columns, ["longitude","lon","경도"])
mag = find_col(df.columns, ["magnitude","mag","mw","규모","규모(m)","m"])
tim = find_col(df.columns, ["time","origin_time","date","발생","시각","utc"])

need = [lat, lon, mag]
if any(c is None for c in need):
    st.error("필수 컬럼(latitude/longitude/magnitude)을 찾을 수 없습니다.\n"
             "가능한 컬럼명 예: latitude/longitude/magnitude/time 또는 위도/경도/규모/발생시각")
    st.stop()

# 정리
df[lat] = pd.to_numeric(df[lat], errors="coerce")
df[lon] = pd.to_numeric(df[lon], errors="coerce")
df[mag] = pd.to_numeric(df[mag], errors="coerce")
if tim:
    # 다양한 포맷 허용
    df[tim] = pd.to_datetime(df[tim], errors="coerce", utc=True)

df = df.dropna(subset=[lat,lon,mag])
df = df[df[mag] >= mag_min].copy()
if len(df) == 0:
    st.warning("조건을 만족하는 데이터가 없습니다.")
    st.stop()

# --- 규모→색상 매핑 ---
def color_for_mag(m):
    if m < 4.0: return [255, 221, 87]   # 연노랑
    if m < 5.0: return [255, 178, 70]   # 주황
    if m < 6.0: return [255, 128, 0]    # 오렌지
    if m < 7.0: return [255, 80, 0]     # 진오렌지
    return [220, 20, 60]                # 빨강

df["_color"] = df[mag].apply(color_for_mag)
df["_size"]  = np.clip((df[mag].fillna(0)+1.0)*size_scale, 1500, 25000)

center_lat = float(df[lat].mean())
center_lon = float(df[lon].mean())

# --- 지도 ---
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
    "html": f"""
    <div style="font-size:14px">
      <b>규모:</b> {{{mag}}}<br/>
      <b>위도:</b> {{{lat}}}, <b>경도:</b> {{{lon}}}<br/>
      {(f"<b>시간:</b> {{{tim}}}" if tim else "")}
    </div>
    """,
    "style": {"backgroundColor":"white","color":"black"}
}

deck = pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=1.7),
    layers=[points],
    tooltip=tooltip
)
st.pydeck_chart(deck, use_container_width=True)

st.markdown("**색상 범례(규모)**  🟨 4.0 미만 · 🟧 4.0–4.9 · 🟠 5.0–5.9 · 🟥 6.0–6.9 · 🧨 7.0+")

# --- 국가별 최신 3건: GeoJSON 폴리곤 내부 판정 ---
geo_path = Path("data/world_countries_simplified.geojson")
if not geo_path.exists():
    st.warning("국가별 최신 3건 기능을 쓰려면 data/world_countries_simplified.geojson 파일을 넣어주세요.")
else:
    if sel_country.strip():
        with open(geo_path, "r", encoding="utf-8") as fp:
            world_geo = json.load(fp)

        def point_in_ring(point, ring):
            x, y = point; inside = False; n = len(ring)
            for i in range(n):
                x1,y1 = ring[i]; x2,y2 = ring[(i+1)%n]
                cond = ((y1>y)!=(y2>y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-12)+x1)
                if cond: inside = not inside
            return inside
        def point_in_poly(point, poly):
            outer = [(p[0],p[1]) for p in poly[0]]
            if not point_in_ring(point, outer): return False
            for hole in poly[1:]:
                if point_in_ring(point, [(p[0],p[1]) for p in hole]): return False
            return True
        def in_multipoly(point, mpoly):
            return any(point_in_poly(point, poly) for poly in mpoly)

        feats = [ft for ft in world_geo["features"]
                 if (ft["properties"].get("ADMIN")==sel_country or ft["properties"].get("name")==sel_country)]
        if not feats:
            st.warning("해당 이름의 국가를 찾지 못했습니다. (예: Japan, Korea, United States)")
        else:
            rows = []
            for _, r in df.iterrows():
                pt = (r[lon], r[lat])
                inside = False
                for ft in feats:
                    g = ft["geometry"]
                    if g["type"]=="Polygon" and point_in_poly(pt, g["coordinates"]):
                        inside = True; break
                    if g["type"]=="MultiPolygon" and in_multipoly(pt, g["coordinates"]):
                        inside = True; break
                if inside: rows.append(r)
            sub = pd.DataFrame(rows)
            if len(sub):
                if tim:
                    sub = sub.sort_values(by=tim, ascending=False).head(3)
                else:
                    sub = sub.sort_values(by=mag, ascending=False).head(3)
                st.subheader(f"🧭 {sel_country} 최신 지진 3건")
                keep = [lat,lon,mag] + ([tim] if tim else [])
                st.dataframe(sub[keep], use_container_width=True)
                st.download_button("⬇️ 최신 3건 CSV 다운로드",
                    data=sub[keep].to_csv(index=False).encode("utf-8"),
                    file_name=f"top3_{sel_country.replace(' ','_')}.csv", mime="text/csv")
            else:
                st.info("해당 국가 경계 내 데이터가 없습니다.")

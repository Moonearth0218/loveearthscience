# pages/02_국내지진분포.py
import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import colorsys

st.set_page_config(page_title="국내 지진 분포", page_icon="🌏", layout="wide")

st.title("🇰🇷 국내 지진 분포 시각화")
st.caption("최근 10년 국내 지진 목록(기상청) 기반 • 위도·경도 위치를 지도에 표시 • 규모별 색상 구분(파랑→빨강) • 원 크기 동일")

# ------------------ 유틸 ------------------
def parse_deg(s):
    """'36.01 N' → 36.01, '128.07 E' → 128.07"""
    if pd.isna(s): return None
    if isinstance(s, (int, float)): return float(s)
    s = str(s).strip()
    parts = s.replace("°", " ").replace("deg", " ").split()
    if not parts: return None
    val = None
    for token in parts:
        try:
            val = float(token); break
        except Exception:
            continue
    if val is None: return None
    s_upper = s.upper()
    return -abs(val) if ("S" in s_upper or "W" in s_upper) else abs(val)

@st.cache_data(show_spinner=False)
def load_data_from_path(path: str):
    xls = pd.ExcelFile(path)
    sheet = xls.sheet_names[0]
    return pd.read_excel(path, sheet_name=sheet)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if {"번호","발생시각","규모"}.issubset(df.columns):
        df = df[~(df["번호"].isna() & df["발생시각"].isna() & df["규모"].isna())]
    if "발생시각" in df.columns:
        df["발생시각"] = pd.to_datetime(df["발생시각"], errors="coerce")
    for c in ["위도","경도"]:
        if c in df.columns: df[c+"_val"] = df[c].apply(parse_deg)
    if "깊이(km)" in df.columns:
        df["깊이(km)"] = pd.to_numeric(df["깊이(km)"], errors="coerce")
    if "규모" in df.columns:
        df["규모"] = pd.to_numeric(df["규모"], errors="coerce")
    if "위치" in df.columns:
        df["위치"] = df["위치"].fillna("미상")

    if {"위도_val","경도_val"}.issubset(df.columns):
        df = df[
            (df["위도_val"].between(32, 39.5, inclusive="both")) &
            (df["경도_val"].between(124, 132.5, inclusive="both"))
        ].copy()

    df = df.dropna(subset=["규모","발생시각","위도_val","경도_val"])
    keep = [c for c in ["번호","발생시각","규모","깊이(km)","최대진도","위치","위도_val","경도_val"] if c in df.columns]
    return df[keep].sort_values("발생시각")

def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            bio = io.BytesIO(uploaded_file.read())
            xls = pd.ExcelFile(bio); sheet = xls.sheet_names[0]
            return clean_dataframe(pd.read_excel(bio, sheet_name=sheet)), "uploaded"
        except Exception as e:
            st.error(f"업로드 파일 읽기 오류: {e}")
            return None, None
    default_path = "data/국내지진목록_10년.xlsx"
    if os.path.exists(default_path):
        try:
            return clean_dataframe(load_data_from_path(default_path)), "default"
        except Exception as e:
            st.warning(f"기본 파일 로드 실패: {e}")
            return None, None
    return None, None

# ------------------ 색상 변환 ------------------
def magnitude_to_color(m, m_min, m_max):
    """규모를 0~1 정규화 후 HSV→RGB 변환 (파랑→빨강)"""
    ratio = (m - m_min) / (m_max - m_min + 1e-9)
    ratio = np.clip(ratio, 0, 1)
    r, g, b = colorsys.hsv_to_rgb(0.66 * (1 - ratio), 1, 1)  # 0.66=파랑, 0=빨강
    return [int(r*255), int(g*255), int(b*255)]

def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*rgb)

def add_vis_columns(df, fixed_radius_m: float):
    vis = df.copy()
    vis["발생시각_str"] = vis["발생시각"].dt.strftime("%Y-%m-%d %H:%M:%S")
    m_min, m_max = vis["규모"].min(), vis["규모"].max()
    vis["color"] = vis["규모"].apply(lambda m: magnitude_to_color(m, m_min, m_max))
    vis["radius_m"] = float(fixed_radius_m)  # 모든 점 동일 크기
    return vis

# ------------------ 업로더 ------------------
st.divider()
st.subheader("데이터 불러오기")
uploaded = st.file_uploader("엑셀 업로드 (.xlsx)", type=["xlsx"], label_visibility="visible")

df_all, data_source = load_data(uploaded)

# ------------------ 사이드바 ------------------
with st.sidebar:
    st.subheader("필터")
    if df_all is not None and len(df_all) > 0:
        min_date = df_all["발생시각"].min().date()
        max_date = df_all["발생시각"].max().date()
        date_range = st.date_input("기간 선택",
                                   value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)
        min_mag = float(df_all["규모"].min())
        max_mag = float(df_all["규모"].max())
        mag_range = st.slider("규모 범위",
                              min_value=round(min_mag,1),
                              max_value=round(max_mag,1),
                              value=(max(2.0, round(min_mag,1)), round(max_mag,1)),
                              step=0.1)
        st.subheader("표시 설정")
        fixed_radius_m = st.slider("표시 원 반지름(미터)", 500, 5000, 2000, step=100)
    else:
        date_range, mag_range, fixed_radius_m = None, None, 2000

st.divider()

# ------------------ 지도 ------------------
if df_all is None or len(df_all) == 0:
    st.warning("데이터가 준비되지 않았습니다. 상단에서 엑셀을 업로드하거나 data 폴더에 파일을 추가하세요.")
else:
    df = df_all.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        df = df[(df["발생시각"] >= start_dt) & (df["발생시각"] < end_dt)]
    if isinstance(mag_range, tuple) and len(mag_range) == 2:
        df = df[df["규모"].between(mag_range[0], mag_range[1], inclusive="both")]

    st.subheader("지진 분포 지도")
    st.write(f"선택된 조건: **{len(df):,}건**  · 데이터 소스: **{data_source}**")

    vis = add_vis_columns(df, fixed_radius_m)
    m_min, m_max = float(vis["규모"].min()), float(vis["규모"].max())

    view_state = pdk.ViewState(latitude=36.5, longitude=127.8, zoom=5.3, pitch=0)
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=vis,
        get_position="[경도_val, 위도_val]",
        get_radius="radius_m",
        get_fill_color="color",
        pickable=True,
        opacity=0.85,
        stroked=True,
        filled=True,
        get_line_color=[30, 30, 30],
        line_width_min_pixels=0.8,
    )
    tooltip = {
        "html": "<b>{발생시각_str}</b><br/>규모: {규모}<br/>깊이: {깊이(km)} km<br/>위치: {위치}",
        "style": {"backgroundColor": "white", "color": "black"}
    }
    deck = pdk.Deck(
        layers=[scatter_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="light"
    )
    st.pydeck_chart(deck, use_container_width=True)

    # ------------------ 색상 범례 (컬러바 바로 아래 눈금 표시) ------------------
    # 1) 그라데이션 CSS 생성
    gradient_stops = []
    for i in range(0, 101, 2):
        ratio = i / 100
        rgb = magnitude_to_color(m_min + ratio*(m_max - m_min), m_min, m_max)
        gradient_stops.append(f"{rgb_to_hex(rgb)} {i}%")
    gradient_css = "linear-gradient(90deg, " + ", ".join(gradient_stops) + ")"

    # 2) 눈금 값/위치 (컬러바 바로 아래 중앙정렬)
    tick_vals = np.linspace(m_min, m_max, 6)  # 6개 눈금
    tick_pos = np.linspace(0, 100, len(tick_vals))  # %
    tick_spans = "".join(
        f'<span style="position:absolute;left:{p:.1f}%;transform:translateX(-50%);'
        f'font-size:12px;color:#333;">M {v:.1f}</span>'
        for v, p in zip(tick_vals, tick_pos)
    )

    legend_html = f"""
    <div style="margin-top:16px;padding:14px;border:1px solid #e0e0e0;border-radius:12px;background:#fff;">
      <div style="font-weight:700;margin-bottom:10px;">규모–색상 안내</div>

      <div style="position:relative;">
        <!-- 컬러바 -->
        <div style="height:18px;border-radius:9px;background:{gradient_css};border:1px solid #bbb;"></div>
        <!-- 눈금 라벨 (컬러바 바로 아래) -->
        <div style="position:relative;height:22px;margin-top:6px;">
          {tick_spans}
        </div>
      </div>

      <div style="font-size:12px;color:#666;margin-top:6px;">
        ※ 원의 크기는 동일하며, 색상만 규모를 나타냅니다. (파랑: 작은 규모 → 빨강: 큰 규모)
      </div>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

    with st.expander("데이터 미리보기"):
        st.dataframe(df.reset_index(drop=True))

st.caption("※ 파일 예시: data/국내지진목록_10년.xlsx  • 업로드 후 자동 시각화")

# pages/02_국내지진분포.py
import os
import io
import pandas as pd
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="국내 지진 분포", page_icon="🌏", layout="wide")

st.title("🇰🇷 국내 지진 분포 시각화")
st.caption("최근 10년 국내 지진 목록(기상청) 기반 • 위도·경도 위치를 지도로 표시 • 규모 구간별 색상 + 히트맵")

# ===============================
#  위경도 파싱 유틸
# ===============================
def parse_deg(s):
    """ '36.01 N' → 36.01, '128.07 E' → 128.07, 남/서반구는 음수 """
    if pd.isna(s):
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    parts = s.replace("°", " ").replace("deg", " ").split()
    if not parts:
        return None
    val = None
    for token in parts:
        try:
            val = float(token)
            break
        except Exception:
            continue
    if val is None:
        return None
    s_upper = s.upper()
    if "S" in s_upper or "W" in s_upper:
        val = -abs(val)
    else:
        val = abs(val)
    return val

@st.cache_data(show_spinner=False)
def load_data_from_path(path: str):
    xls = pd.ExcelFile(path)
    sheet = xls.sheet_names[0]  # 첫 시트 사용
    df = pd.read_excel(path, sheet_name=sheet)
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # 완전 빈 첫 행 제거
    if {"번호", "발생시각", "규모"}.issubset(df.columns):
        df = df[~(df["번호"].isna() & df["발생시각"].isna() & df["규모"].isna())]

    # 타입 정리
    if "발생시각" in df.columns:
        df["발생시각"] = pd.to_datetime(df["발생시각"], errors="coerce")
    for col in ["위도", "경도"]:
        if col in df.columns:
            df[col + "_val"] = df[col].apply(parse_deg)
    if "깊이(km)" in df.columns:
        df["깊이(km)"] = pd.to_numeric(df["깊이(km)"], errors="coerce")
    if "규모" in df.columns:
        df["규모"] = pd.to_numeric(df["규모"], errors="coerce")
    if "위치" in df.columns:
        df["위치"] = df["위치"].fillna("미상")

    # 한반도 대략 범위 필터
    if {"위도_val", "경도_val"}.issubset(df.columns):
        df = df[
            (df["위도_val"].between(32, 39.5, inclusive="both")) &
            (df["경도_val"].between(124, 132.5, inclusive="both"))
        ].copy()

    # 필수 결측 제거
    need_cols = ["규모", "발생시각", "위도_val", "경도_val"]
    df = df.dropna(subset=[c for c in need_cols if c in df.columns])

    keep_cols = [c for c in ["번호", "발생시각", "규모", "깊이(km)", "최대진도", "위치", "위도_val", "경도_val"] if c in df.columns]
    return df[keep_cols].sort_values("발생시각")

def load_data(uploaded_file):
    """
    업로더(본문)로 올린 파일이 있으면 그걸 사용하고,
    없으면 data/국내지진목록_10년.xlsx 를 시도.
    """
    # 1) 업로드 파일 우선
    if uploaded_file is not None:
        try:
            bio = io.BytesIO(uploaded_file.read())
            xls = pd.ExcelFile(bio)
            sheet = xls.sheet_names[0]
            df = pd.read_excel(bio, sheet_name=sheet)
            return clean_dataframe(df), "uploaded"
        except Exception as e:
            st.error(f"업로드 파일을 읽는 중 오류: {e}")
            return None, None

    # 2) 기본 경로
    default_path = "data/국내지진목록_10년.xlsx"
    if os.path.exists(default_path):
        try:
            df = load_data_from_path(default_path)
            return clean_dataframe(df), "default"
        except Exception as e:
            st.warning(f"기본 경로 파일 로드 실패: {e}")
            return None, None

    return None, None

# ===============================
#  시각화용 컬럼/설정
# ===============================
BIN_DEFS = [
    (2.0, 2.9, [173, 216, 230]),  # light blue
    (3.0, 3.9, [135, 206, 250]),  # sky blue
    (4.0, 4.9, [255, 165, 0]),    # orange
    (5.0, 5.9, [255, 99, 71]),    # tomato
    (6.0, 99.0, [178, 34, 34]),   # firebrick
]

def mag_to_color(m):
    if pd.isna(m):
        return [160, 160, 160]
    for lo, hi, rgb in BIN_DEFS:
        if lo <= m <= hi:
            return rgb
    return [160, 160, 160]

def add_vis_columns(df):
    vis = df.copy()
    vis["발생시각_str"] = vis["발생시각"].dt.strftime("%Y-%m-%d %H:%M:%S")
    base = float(vis["규모"].min())
    vis["radius_m"] = (vis["규모"] - base + 1.0) * 2500
    vis["color"] = vis["규모"].apply(mag_to_color)
    vis["weight"] = vis["규모"].clip(lower=0)
    return vis

def legend_html():
    chips = "".join(
        f"""
        <div style="display:flex;align-items:center;gap:8px;margin:2px 0;">
          <span style="display:inline-block;width:14px;height:14px;border-radius:50%;
                       background: rgb({c[0]},{c[1]},{c[2]});border:1px solid #999;"></span>
          <span style="font-size:12px;">M {lo:.1f}–{hi:.1f}</span>
        </div>
        """
        for (lo, hi, c) in BIN_DEFS
    )
    return f"""
    <div style="padding:10px 12px;border:1px solid #ddd;border-radius:8px;background:#fff;">
      <div style="font-weight:600;margin-bottom:6px;">규모 구간 범례</div>
      {chips}
    </div>
    """

# ===============================
#  본문 상단: 파일 업로더(오른쪽 흰 영역)
# ===============================
st.divider()
st.subheader("데이터 불러오기")
left, right = st.columns([2, 3])
with left:
    st.markdown("엑셀 파일을 업로드하거나, 프로젝트 루트의 `data/국내지진목록_10년.xlsx`를 사용합니다.")
with right:
    uploaded = st.file_uploader("엑셀 업로드 (.xlsx)", type=["xlsx"], label_visibility="visible")

df_all, data_source = load_data(uploaded)

# ===============================
#  사이드바: 필터 (데이터 로드 이후 표시)
# ===============================
with st.sidebar:
    st.subheader("필터")
    if df_all is None or len(df_all) == 0:
        st.info("데이터를 먼저 불러오세요.")
        date_range = None
        mag_range = None
    else:
        min_date = df_all["발생시각"].min().date()
        max_date = df_all["발생시각"].max().date()
        date_range = st.date_input(
            "기간 선택",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        min_mag = float(df_all["규모"].min())
        max_mag = float(df_all["규모"].max())
        mag_range = st.slider(
            "규모 범위",
            min_value=round(min_mag, 1),
            max_value=round(max_mag, 1),
            value=(max(2.0, round(min_mag, 1)), round(max_mag, 1)),
            step=0.1
        )

st.divider()

# ===============================
#  지도 섹션
# ===============================
st.subheader("지도에서 보기")
col1, col2 = st.columns([1, 2], vertical_alignment="center")
with col1:
    st.markdown("**1) 버튼을 눌러 지도를 생성하세요.**")
    draw_btn = st.button("🗺️ 국내 지진 분포 보기", use_container_width=True)
with col2:
    st.markdown(
        """
        - **원 색상**: 규모 구간별 색상 (아래 범례 참조)  
        - **원 크기**: 규모에 비례  
        - **히트맵**: 같은 지역에 쌓이는 패턴 시각화  
        - **툴팁**: 발생시각 · 규모 · 깊이 · 위치  
        """
    )

st.markdown(legend_html(), unsafe_allow_html=True)

if df_all is None or len(df_all) == 0:
    st.warning("데이터가 준비되지 않았습니다. 상단에서 엑셀을 업로드하거나, data/국내지진목록_10년.xlsx 를 배치하세요.")
else:
    # 필터 적용
    df = df_all.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        df = df[(df["발생시각"] >= start_dt) & (df["발생시각"] < end_dt)]
    if isinstance(mag_range, tuple) and len(mag_range) == 2:
        df = df[df["규모"].between(mag_range[0], mag_range[1], inclusive="both")]

    st.write(f"선택된 조건: **{len(df):,}건**"
             + ("" if data_source is None else f"  · 데이터 소스: **{data_source}**"))

    if draw_btn:
        vis = add_vis_columns(df)
        view_state = pdk.ViewState(latitude=36.5, longitude=127.8, zoom=5.3, pitch=0)

        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=vis,
            get_position="[경도_val, 위도_val]",
            get_radius="radius_m",
            get_fill_color="color",
            get_line_color=[60, 60, 60],
            pickable=True,
            opacity=0.35,
            stroked=True,
            filled=True,
            line_width_min_pixels=1,
        )

        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            data=vis,
            get_position="[경도_val, 위도_val]",
            get_weight="weight",
            aggregation='"MEAN"',
            radiusPixels=50,
            intensity=1.0,
            threshold=0.05
        )

        tooltip = {
            "html": "<b>{발생시각_str}</b><br/>규모: {규모}<br/>깊이: {깊이(km)} km<br/>위치: {위치}",
            "style": {"backgroundColor": "white", "color": "black"}
        }

        deck = pdk.Deck(
            layers=[heatmap_layer, scatter_layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="light"
        )
        st.pydeck_chart(deck, use_container_width=True)

        with st.expander("데이터 미리보기"):
            st.dataframe(df.reset_index(drop=True))

# 푸터
st.caption("※ 파일 구조 예시: 프로젝트 루트/data/국내지진목록_10년.xlsx  • 필요시 상단에서 업로드 가능")

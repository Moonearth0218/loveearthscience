import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io
import plotly.express as px
import plotly.colors as plc  # 색상 스케일 샘플링용

# --------------------
# 기본 설정
# --------------------
st.set_page_config(page_title="🌍 지진 데이터 월드맵", page_icon="🌍", layout="wide")

st.title("🌍 지진 데이터 월드맵")
st.caption("KMA 국외지진목록 데이터를 기반으로 규모(M)별 전세계 지진을 시각화합니다. (Plotly)")

st.markdown(
    """
    **사용법**  
    1) 아래에서 파일을 선택하거나, 기본 파일명을 그대로 사용합니다.  
    2) 왼쪽 사이드바에서 기간/규모/깊이/지역 필터를 조정하세요.  
    3) 지도를 확대/이동하면 상세 위치와 정보를 툴팁으로 확인할 수 있어요.  
    """
)

# --------------------
# 데이터 로더
# --------------------
DEFAULT_FILE = "국외지진목록_2015-01-01_2025-09-29.xls"

def read_kma_xls_like(file_obj_or_path):
    """
    KMA 국외지진목록 .xls은 실제로 HTML 테이블인 경우가 많음.
    pandas.read_html로 읽어 1번째 테이블을 반환.
    """
    try:
        tables = pd.read_html(file_obj_or_path)  # lxml 필요
        df = tables[0]
        return df
    except Exception as e:
        raise RuntimeError(f"HTML 테이블 파싱 실패: {e}")

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # 문자열 공백 정리
    df.columns = [str(c).strip() for c in df.columns]
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # 컬럼 탐지 유틸
    def find_col(cols, keywords):
        cols_low = {c.lower(): c for c in cols}
        for c in cols:
            lc = c.lower()
            for k in keywords:
                if k in lc:
                    return cols_low[lc]
        return None

    cols = df.columns.tolist()

    # 흔한 컬럼명 후보(한국어/영문 혼합 지원)
    col_time_utc = find_col(cols, ["발생일시(utc)", "utc"])
    col_time_kst = find_col(cols, ["발생일시(kst)", "kst"])
    col_time_any = find_col(cols, ["발생일시", "date", "time", "일시"])
    col_lat = find_col(cols, ["위도", "latitude", "lat"])
    col_lon = find_col(cols, ["경도", "longitude", "lon"])
    col_depth = find_col(cols, ["깊이", "depth"])
    col_mag = find_col(cols, ["규모", "magnitude", "mag"])
    col_place = find_col(cols, ["위치", "지역", "장소", "place", "location"])
    col_remark = find_col(cols, ["비고", "remark", "참고"])

    out = pd.DataFrame()

    # 시각: UTC > KST > ANY 순으로 우선
    time_col = col_time_utc or col_time_kst or col_time_any
    if time_col:
        out["time"] = pd.to_datetime(df[time_col], errors="coerce")

    # 숫자형 변환(쉼표 제거)
    def to_num(s):
        return pd.to_numeric(pd.Series(s).astype(str).str.replace(",", ""), errors="coerce")

    if col_lat:   out["latitude"]  = to_num(df[col_lat])
    if col_lon:   out["longitude"] = to_num(df[col_lon])
    if col_depth: out["depth_km"]  = to_num(df[col_depth])
    if col_mag:   out["magnitude"] = to_num(df[col_mag])
    if col_place: out["place"]     = df[col_place].astype(str)
    if col_remark:out["remark"]    = df[col_remark].astype(str)

    # 유효 범위 필터(위도/경도 기본 품질 확보)
    if "latitude" in out and "longitude" in out:
        out = out[(out["latitude"].between(-90, 90)) & (out["longitude"].between(-180, 180))]

    # 정렬
    if "time" in out:
        out = out.sort_values("time").reset_index(drop=True)

    return out

# --------------------
# 파일 입력
# --------------------
left, right = st.columns([1, 1])
with left:
    st.subheader("📁 데이터 선택")
    up = st.file_uploader("국외지진목록(.xls / HTML 테이블 형식) 파일 업로드", type=["xls", "html", "htm"])
    use_default = st.toggle(f"기본 파일명 사용: `{DEFAULT_FILE}`", value=True)

# 데이터 읽기
df_raw = None
error_msg = None

try:
    if up is not None:
        # 업로더 파일을 그대로 read_html에 전달
        content = io.BytesIO(up.read())
        df_raw = read_kma_xls_like(content)
    else:
        # 업로더가 없으면 기본 파일 사용 시도
        if use_default and Path(DEFAULT_FILE).exists():
            df_raw = read_kma_xls_like(DEFAULT_FILE)
        elif use_default:
            st.info(f"기본 파일 `{DEFAULT_FILE}` 을(를) 찾을 수 없습니다. 파일을 업로드하세요.")
except Exception as e:
    error_msg = str(e)

if error_msg:
    st.error(error_msg)

if df_raw is not None and not df_raw.empty:
    # 클린업
    df = clean_dataframe(df_raw)

    if df.empty or {"latitude", "longitude"}.issubset(df.columns) is False:
        st.error("위도/경도 컬럼을 해석하지 못했습니다. 원본 테이블의 위도/경도 표기를 확인해주세요.")
        st.stop()

    # --------------------
    # 사이드바 필터
    # --------------------
    with st.sidebar:
        st.header("🧭 필터")
        # 날짜 필터
        if "time" in df.columns and df["time"].notna().any():
            tmin = pd.to_datetime(df["time"].min())
            tmax = pd.to_datetime(df["time"].max())
            date_range = st.date_input(
                "기간 선택",
                value=(tmin.date(), tmax.date()),
                min_value=tmin.date(), max_value=tmax.date()
            )
        else:
            date_range = None

        # 규모 필터
        if "magnitude" in df.columns and df["magnitude"].notna().any():
            mag_min = float(np.nanmin(df["magnitude"]))
            mag_max = float(np.nanmax(df["magnitude"]))
            m_lo, m_hi = st.slider("규모(M) 범위", min_value=float(np.floor(mag_min)), max_value=float(np.ceil(mag_max)),
                                   value=(float(np.floor(mag_min)), float(np.ceil(mag_max))), step=0.1)
        else:
            m_lo, m_hi = None, None

        # 깊이 필터
        if "depth_km" in df.columns and df["depth_km"].notna().any():
            dmin = float(np.nanmin(df["depth_km"]))
            dmax = float(np.nanmax(df["depth_km"]))
            dep_lo, dep_hi = st.slider("깊이(km) 범위", min_value=float(max(0.0, np.floor(dmin))),
                                       max_value=float(np.ceil(dmax)),
                                       value=(float(max(0.0, np.floor(dmin))), float(np.ceil(dmax))), step=1.0)
        else:
            dep_lo, dep_hi = None, None

        # 지역 텍스트 검색
        place_query = st.text_input("지역/위치 키워드 🔎", value="").strip()

    # --------------------
    # 필터 적용
    # --------------------
    df_f = df.copy()

    # 날짜
    if date_range and "time" in df_f.columns and df_f["time"].notna().any():
        start_dt = pd.to_datetime(pd.Timestamp(date_range[0]))
        end_dt = pd.to_datetime(pd.Timestamp(date_range[1])) + pd.Timedelta(days=1)  # inclusive
        df_f = df_f[(df_f["time"] >= start_dt) & (df_f["time"] < end_dt)]

    # 규모
    if m_lo is not None and m_hi is not None and "magnitude" in df_f.columns:
        df_f = df_f[df_f["magnitude"].between(m_lo, m_hi)]

    # 깊이
    if dep_lo is not None and dep_hi is not None and "depth_km" in df_f.columns:
        df_f = df_f[df_f["depth_km"].between(dep_lo, dep_hi)]

    # 지역 검색
    if place_query and "place" in df_f.columns:
        df_f = df_f[df_f["place"].str.contains(place_query, case=False, na=False)]

    # --------------------
    # 규모 '정수 구간' 색상 매핑 (파랑 → 빨강)
    # --------------------
    if "magnitude" in df_f.columns and df_f["magnitude"].notna().any():
        # 정수 하한(예: 2.3 -> 2)을 구간 ID로 사용
        mag_floor = np.floor(df_f["magnitude"]).astype("Int64")
        # 범례 라벨: "2.0–2.9" 형태
        df_f["mag_bin_label"] = mag_floor.map(lambda v: f"{int(v)}.0–{int(v)}.9" if pd.notna(v) else np.nan)

        # 실제 등장한 구간만 추출 (작은→큰 순)
        unique_bins = sorted(mag_floor.dropna().unique().tolist())
        labels_order = [f"{int(v)}.0–{int(v)}.9" for v in unique_bins]

        # 파랑→빨강 색 스케일에서 구간 수만큼 균등 샘플링
        # Bluered는 0:파랑, 1:빨강 이므로 작은 M일수록 파랑, 큰 M일수록 빨강
        base_scale = px.colors.sequential.Bluered
        # 샘플 함수: positions 0~1 사이 선형
        positions = np.linspace(0, 1, num=len(labels_order))
        # plotly에 내장된 유틸이 없어도 base_scale를 직접 보간 없이 균등 선택
        # 구간 수가 base_scale 길이보다 크면 반복 샘플링
        def pick_color(pos):
            # pos ∈ [0,1], base_scale 길이에 맞춰 인덱스 선택
            idx = int(round(pos * (len(base_scale) - 1)))
            return base_scale[idx]
        color_list = [pick_color(p) for p in positions]
        color_map = {label: color_list[i] for i, label in enumerate(labels_order)}
    else:
        df_f["mag_bin_label"] = np.nan
        labels_order, color_map = [], {}

    # --------------------
    # 상단 KPI
    # --------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("표시 건수", f"{len(df_f):,}")
    if "magnitude" in df_f.columns and df_f["magnitude"].notna().any():
        k2.metric("평균 규모", f"{df_f['magnitude'].mean():.2f}")
        k3.metric("최대 규모", f"{df_f['magnitude'].max():.1f}")
    else:
        k2.metric("평균 규모", "-")
        k3.metric("최대 규모", "-")
    if "depth_km" in df_f.columns and df_f["depth_km"].notna().any():
        k4.metric("평균 깊이(km)", f"{df_f['depth_km'].mean():.0f}")
    else:
        k4.metric("평균 깊이(km)", "-")

    # --------------------
    # 지도 시각화 (Plotly) : 정수 구간별 '이산 색상' 적용
    # --------------------
    st.subheader("🗺️ 전세계 지진 분포 (규모 정수 구간별 이산 색)")
    hover_cols = []
    if "time" in df_f.columns: hover_cols.append("time")
    if "place" in df_f.columns: hover_cols.append("place")
    if "depth_km" in df_f.columns: hover_cols.append("depth_km")
    if "magnitude" in df_f.columns: hover_cols.append("magnitude")

    size_col = "magnitude" if "magnitude" in df_f.columns else None

    fig = px.scatter_geo(
        df_f,
        lat="latitude",
        lon="longitude",
        size=size_col,
        color="mag_bin_label",                   # 이산(범주) 색상
        color_discrete_map=color_map,            # 구간→색 매핑
        category_orders={"mag_bin_label": labels_order},  # 범례 순서(작은→큰)
        size_max=16,
        opacity=0.8,
        hover_data=hover_cols,
        projection="natural earth",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        legend_title_text="규모 구간(M)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --------------------
    # 데이터 미리보기
    # --------------------
    with st.expander("📄 데이터 미리보기 (필터 적용 후)"):
        st.dataframe(df_f.head(100), use_container_width=True)
else:
    st.info("왼쪽에서 파일을 업로드하거나, 기본 파일이 있을 경우 토글을 켜서 불러오세요.")

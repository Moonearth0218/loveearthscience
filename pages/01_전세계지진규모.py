import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io
import plotly.express as px

# --------------------
# 기본 설정
# --------------------
st.set_page_config(page_title="🗺️ 전세계 지진 규모", page_icon="🗺️", layout="wide")

st.title("🗺️ 전세계 지진 규모 분석")
st.caption("KMA 국외지진목록 데이터를 업로드하여 규모(M) 정수 구간별 색상으로 시각화합니다.")

DEFAULT_FILE = "국외지진목록_2015-01-01_2025-09-29.xls"

@st.cache_data(show_spinner=False)
def read_kma_xls_like(file_obj_or_path):
    """
    KMA 국외지진목록 .xls은 실제로 HTML 테이블인 경우가 많음.
    - lxml 파서만 사용(flavor='lxml') → html5lib 의존성 제거
    - 여러 테이블이 있으면 위도/경도/규모/깊이 컬럼 포함 여부를 기준으로 가장 적합한 테이블 선택
    """
    try:
        tables = pd.read_html(file_obj_or_path, flavor="lxml")
        if len(tables) == 0:
            raise RuntimeError("HTML에서 표를 찾지 못했습니다.")

        def score_table(df):
            cols = [str(c).lower() for c in df.columns]
            score = 0
            if any(("위도" in c) or ("lat" in c) for c in cols): score += 2
            if any(("경도" in c) or ("lon" in c) or ("lng" in c) for c in cols): score += 2
            if any(("규모" in c) or ("mag" in c) for c in cols): score += 1
            if any(("깊이" in c) or ("depth" in c) for c in cols): score += 1
            if any(("발생일시" in c) or ("date" in c) or ("time" in c) for c in cols): score += 1
            return score

        tables_scored = sorted(tables, key=score_table, reverse=True)
        return tables_scored[0]

    except Exception as e:
        raise RuntimeError(f"HTML 테이블 파싱 실패(lxml): {e}")

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    def find_col(cols, keywords):
        for c in cols:
            lc = c.lower()
            if any(k in lc for k in keywords):
                return c
        return None

    cols = df.columns.tolist()
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
    time_col = col_time_utc or col_time_kst or col_time_any
    if time_col:
        out["time"] = pd.to_datetime(df[time_col], errors="coerce")

    def to_num(s):
        return pd.to_numeric(pd.Series(s).astype(str).str.replace(",", ""), errors="coerce")

    if col_lat:   out["latitude"]  = to_num(df[col_lat])
    if col_lon:   out["longitude"] = to_num(df[col_lon])
    if col_depth: out["depth_km"]  = to_num(df[col_depth])
    if col_mag:   out["magnitude"] = to_num(df[col_mag])
    if col_place: out["place"]     = df[col_place].astype(str)
    if col_remark:out["remark"]    = df[col_remark].astype(str)

    if "latitude" in out and "longitude" in out:
        out = out[(out["latitude"].between(-90, 90)) & (out["longitude"].between(-180, 180))]
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

df_raw = None
if up is not None:
    try:
        content = io.BytesIO(up.read())
        df_raw = read_kma_xls_like(content)
    except Exception as e:
        st.error("파일을 읽는 중 오류가 발생했습니다.")
        st.exception(e)
else:
    if use_default and Path(DEFAULT_FILE).exists():
        try:
            df_raw = read_kma_xls_like(DEFAULT_FILE)
        except Exception as e:
            st.error("기본 파일을 읽는 중 오류가 발생했습니다.")
            st.exception(e)
    elif use_default:
        st.info(f"기본 파일 `{DEFAULT_FILE}` 을(를) 찾을 수 없습니다. 파일을 업로드하세요.")

if df_raw is not None and not df_raw.empty:
    df = clean_dataframe(df_raw)

    if df.empty or {"latitude", "longitude"}.issubset(df.columns) is False:
        st.error("위도/경도 컬럼을 해석하지 못했습니다. 원본 테이블의 위도/경도 표기를 확인해주세요.")
        st.stop()

    # 사이드바 필터
    with st.sidebar:
        st.header("🧭 필터")
        if "time" in df.columns and df["time"].notna().any():
            tmin = pd.to_datetime(df["time"].min())
            tmax = pd.to_datetime(df["time"].max())
            date_range = st.date_input("기간 선택",
                value=(tmin.date(), tmax.date()),
                min_value=tmin.date(), max_value=tmax.date()
            )
        else:
            date_range = None

        if "magnitude" in df.columns and df["magnitude"].notna().any():
            mag_min = float(np.nanmin(df["magnitude"]))
            mag_max = float(np.nanmax(df["magnitude"]))
            m_lo, m_hi = st.slider("규모(M) 범위",
                min_value=float(np.floor(mag_min)),
                max_value=float(np.ceil(mag_max)),
                value=(float(np.floor(mag_min)), float(np.ceil(mag_max))),
                step=0.1
            )
        else:
            m_lo, m_hi = None, None

        if "depth_km" in df.columns and df["depth_km"].notna().any():
            dmin = float(np.nanmin(df["depth_km"]))
            dmax = float(np.nanmax(df["depth_km"]))
            dep_lo, dep_hi = st.slider("깊이(km) 범위",
                min_value=float(max(0.0, np.floor(dmin))),
                max_value=float(np.ceil(dmax)),
                value=(float(max(0.0, np.floor(dmin))), float(np.ceil(dmax))),
                step=1.0
            )
        else:
            dep_lo, dep_hi = None, None

        place_query = st.text_input("지역/위치 키워드 🔎", value="").strip()

    # 필터 적용
    df_f = df.copy()
    if date_range and "time" in df_f.columns and df_f["time"].notna().any():
        start_dt = pd.to_datetime(pd.Timestamp(date_range[0]))
        end_dt = pd.to_datetime(pd.Timestamp(date_range[1])) + pd.Timedelta(days=1)
        df_f = df_f[(df_f["time"] >= start_dt) & (df_f["time"] < end_dt)]
    if m_lo is not None and m_hi is not None and "magnitude" in df_f.columns:
        df_f = df_f[df_f["magnitude"].between(m_lo, m_hi)]
    if dep_lo is not None and dep_hi is not None and "depth_km" in df_f.columns:
        df_f = df_f[df_f["depth_km"].between(dep_lo, dep_hi)]
    if place_query and "place" in df_f.columns:
        df_f = df_f[df_f["place"].str.contains(place_query, case=False, na=False)]

    # 규모 정수 구간 라벨 & 색상 매핑
    if "magnitude" in df_f.columns and df_f["magnitude"].notna().any():
        mag_floor = np.floor(df_f["magnitude"]).astype("Int64")
        df_f["mag_bin_label"] = mag_floor.map(lambda v: f"{int(v)}.0–{int(v)}.9" if pd.notna(v) else np.nan)
        unique_bins = sorted(mag_floor.dropna().unique().tolist())
        labels_order = [f"{int(v)}.0–{int(v)}.9" for v in unique_bins]

        base_scale = px.colors.sequential.Bluered
        def pick_color(pos):
            idx = int(round(pos * (len(base_scale) - 1)))
            return base_scale[idx]
        positions = np.linspace(0, 1, num=len(labels_order)) if labels_order else []
        color_list = [pick_color(p) for p in positions]
        color_map = {label: color_list[i] for i, label in enumerate(labels_order)}
    else:
        df_f["mag_bin_label"] = np.nan
        labels_order, color_map = [], {}

    # KPI
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

    # 지도
    st.subheader("🌍 규모 정수 구간별 색상 지진 지도")
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
        color="mag_bin_label",
        color_discrete_map=color_map,
        category_orders={"mag_bin_label": labels_order},
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

    with st.expander("📄 데이터 미리보기 (필터 적용 후)"):
        st.dataframe(df_f.head(100), use_container_width=True)
else:
    st.info("왼쪽에서 파일을 업로드하거나, 기본 파일이 있을 경우 토글을 켜서 불러오세요.")

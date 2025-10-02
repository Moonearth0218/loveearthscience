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

# ---------- 새 로더: 형식 자동 분기 ----------
@st.cache_data(show_spinner=False)
def load_quakes(file_bytes: bytes, filename: str = "uploaded") -> pd.DataFrame:
    """
    업로드된 바이트를 검사해 형식별로 안전하게 읽는다.
    지원: HTML(UTF-8/CP949/EUC-KR), XLS(legacy), XLSX, CSV
    """
    b = file_bytes
    head = b[:64].lstrip()

    # 1) XLSX (ZIP 시그니처: PK)
    if head.startswith(b"PK"):
        return pd.read_excel(io.BytesIO(b), engine="openpyxl")

    # 2) Legacy XLS (OLE2 시그니처: D0 CF 11 E0 A1 B1 1A E1)
    if head.startswith(b"\xD0\xCF\x11\xE0"):
        # xlrd는 xls만 지원
        return pd.read_excel(io.BytesIO(b), engine="xlrd")

    # 3) HTML (.xls이지만 사실 HTML 테이블인 경우)
    if head.startswith(b"<!DOCTYPE") or head.startswith(b"<html") or b"<table" in b[:4096].lower():
        # 인코딩 추정 없이 순차 시도 (추가 라이브러리 없이 처리)
        for enc in ["utf-8", "cp949", "euc-kr"]:
            try:
                text = b.decode(enc, errors="strict")
                tables = pd.read_html(io.StringIO(text), flavor="lxml")
                if len(tables):
                    return tables[0]
            except Exception:
                continue
        # 느슨 모드(깨진 글자는 무시)
        try:
            text = b.decode("cp949", errors="ignore")
            tables = pd.read_html(io.StringIO(text), flavor="lxml")
            if len(tables):
                return tables[0]
        except Exception as e:
            raise RuntimeError(f"HTML 테이블 파싱 실패(lxml, 인코딩): {e}")

    # 4) CSV 가능성 (쉼표/탭 자동 추정)
    try:
        return pd.read_csv(io.BytesIO(b))
    except Exception:
        pass
    try:
        return pd.read_csv(io.BytesIO(b), sep="\t")
    except Exception:
        pass

    raise RuntimeError(
        f"알 수 없는 형식입니다. 파일명: {filename} (선두 바이트: {head[:16]!r})"
    )

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
    up = st.file_uploader("국외지진목록 파일 업로드 (.xls, .xlsx, .html, .htm, .csv)", type=["xls", "xlsx", "html", "htm", "csv"])
    use_default = st.toggle(f"기본 파일명 사용: `{DEFAULT_FILE}`", value=True)

# 데이터 읽기
df_raw = None
if up is not None:
    try:
        buf = up.read()
        df_raw = load_quakes(buf, filename=up.name)
    except Exception as e:
        st.error("파일을 읽는 중 오류가 발생했습니다.")
        st.exception(e)
else:
    if use_default and Path(DEFAULT_FILE).exists():
        try:
            with open(DEFAULT_FILE, "rb") as f:
                df_raw = load_quakes(f.read(), filename=DEFAULT_FILE)
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

    # -------- 사이드바 필터 --------
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

    # -------- 필터 적용 --------
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

    # -------- 규모 정수 구간 라벨 & 색상 --------
    if "magnitude" in df_f.columns and df_f["magnitude"].notna().any():
        mag_floor = np.floor(df_f["magnitude"]).astype("Int64")
        df_f["mag_bin_label"] = mag_floor.map(lambda v: f"{int(v)}.0–{int(v)}.9" if pd.notna(v) else np.nan)
        unique_bins = sorted(mag_floor.dropna().unique().tolist())
        labels_order = [f"{int(v)}.0–{int(v)}.9" for v in unique_bins]

        base_scale = px.colors.sequential.Bluered  # 파랑→빨강
        def pick_color(pos):
            idx = int(round(pos * (len(base_scale) - 1)))
            return base_scale[idx]
        positions = np.linspace(0, 1, num=len(labels_order)) if labels_order else []
        color_list = [pick_color(p) for p in positions]
        color_map = {label: color_list[i] for i, label in enumerate(labels_order)}
    else:
        df_f["mag_bin_label"] = np.nan
        labels_order, color_map = [], {}

    # -------- KPI --------
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

    # -------- 지도 --------
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
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="규모 구간(M)")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 데이터 미리보기 (필터 적용 후)"):
        st.dataframe(df_f.head(100), use_container_width=True)

else:
    st.info("왼쪽에서 파일을 업로드하거나, 기본 파일이 있을 경우 토글을 켜서 불러오세요.")

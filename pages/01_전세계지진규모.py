import io
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# 기본 설정
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🌍 전세계 지진 규모 지도", page_icon="🌍", layout="wide")
st.title("🌍 전세계 지진 규모 지도")
st.caption("지진의 위도·경도 위치에 규모(M) 정수 구간별 색상을 적용해 전세계 지도를 시각화합니다. (작을수록 파랑, 클수록 빨강)")

# ──────────────────────────────────────────────────────────────────────────────
# 유틸: 좌표/숫자/라벨 처리
# ──────────────────────────────────────────────────────────────────────────────
def parse_coord(series: pd.Series, kind: str) -> pd.Series:
    """
    위도/경도 열에 '24.72 N', '66.67 W' 같은 표기가 있어도 숫자로 변환.
    kind='lat'이면 S에 음수, 'lon'이면 W에 음수.
    """
    s = series.astype(str).str.strip()
    # 숫자만 추출
    num = pd.to_numeric(s.str.extract(r'([-+]?\d+(?:\.\d+)?)')[0], errors="coerce")
    if kind == "lat":
        return num.mask(s.str.contains(r"[Ss]"), -num)
    else:
        return num.mask(s.str.contains(r"[Ww]"), -num)

def to_num(series: pd.Series) -> pd.Series:
    """쉼표 등의 문자를 제거하고 float로 변환"""
    return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")

def make_mag_bin_label(m: pd.Series) -> pd.Series:
    """
    규모를 1.0 단위 구간으로 라벨링.
    0→'0.0–0.9', 1→'1.0–1.9', …, 9→'9.0–9.9', 10 이상→'10.0'
    음수나 NaN은 NaN 처리.
    """
    mf = np.floor(m).astype("Int64")
    mf = mf.clip(lower=0, upper=10)
    def lab(v):
        if pd.isna(v): return np.nan
        v = int(v)
        return f"{v}.0–{v}.9" if v < 10 else "10.0"
    return mf.map(lab)

def build_color_map(labels_order):
    """
    작은 구간→파랑, 큰 구간→빨강.
    Plotly의 Bluered(파→빨)에서 구간 수만큼 균등 샘플링해 이산 색상으로 매핑.
    """
    base = px.colors.sequential.Bluered  # 0:파랑 → 1:빨강
    def pick(pos):
        idx = int(round(pos * (len(base) - 1)))
        return base[idx]
    positions = np.linspace(0, 1, num=len(labels_order)) if labels_order else []
    colors = [pick(p) for p in positions]
    return {label: colors[i] for i, label in enumerate(labels_order)}

# ──────────────────────────────────────────────────────────────────────────────
# 로더: 다양한 포맷(.xlsx/.xls/.html/.htm/.csv) 자동 판별
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_quakes(file_bytes: bytes, filename: str = "uploaded") -> pd.DataFrame:
    b = file_bytes or b""
    head = b[:64].lstrip()

    # 1) XLSX (ZIP 시그니처 PK)
    if head.startswith(b"PK"):
        return pd.read_excel(io.BytesIO(b), engine="openpyxl")

    # 2) 구형 XLS (OLE 시그니처)
    if head.startswith(b"\xD0\xCF\x11\xE0"):
        return pd.read_excel(io.BytesIO(b), engine="xlrd")

    # 3) HTML (확장자만 xls여도 HTML 테이블일 수 있음)
    looks_html = head.startswith(b"<!DOCTYPE") or head.startswith(b"<html") or (b"<table" in b[:8192].lower())
    if looks_html:
        # 인코딩 후보별로 html5lib → lxml 순서 시도
        for enc in ["utf-8", "cp949", "euc-kr"]:
            try:
                text = b.decode(enc, errors="strict")
                for flavor in ["html5lib", "lxml"]:
                    try:
                        tables = pd.read_html(io.StringIO(text), flavor=flavor)
                        if len(tables):
                            return tables[0]
                    except Exception:
                        pass
            except Exception:
                pass
        # 마지막 시도: BeautifulSoup로 첫 번째 table만 추출
        try:
            from bs4 import BeautifulSoup
            text = b.decode("cp949", errors="ignore")
            soup = BeautifulSoup(text, "html.parser")
            table = soup.find("table")
            if table:
                tables = pd.read_html(io.StringIO(str(table)), flavor="lxml")
                if len(tables):
                    return tables[0]
        except Exception:
            pass
        raise RuntimeError("HTML 테이블을 찾지 못했습니다. (html5lib/lxml/bs4 실패)")

    # 4) CSV 추정 (, / \t / ;)
    for kwargs in [dict(), dict(sep="\t"), dict(sep=";")]:
        try:
            return pd.read_csv(io.BytesIO(b), **kwargs)
        except Exception:
            pass

    raise RuntimeError(f"알 수 없는 파일 형식입니다: {filename}")

# ──────────────────────────────────────────────────────────────────────────────
# 클린업: 컬럼 자동 감지 + 수동 매핑 폴백
# ──────────────────────────────────────────────────────────────────────────────
def auto_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    def find_col(cols, keys):
        for c in cols:
            lc = c.lower()
            if any(k in lc for k in keys):
                return c
        return None

    cols = df.columns.tolist()
    col_time = find_col(cols, ["발생시각", "발생일시", "date", "time", "일시"])
    col_mag  = find_col(cols, ["규모", "magnitude", "mag"])
    col_dep  = find_col(cols, ["깊이", "depth"])
    col_lat  = find_col(cols, ["위도", "latitude", "lat"])
    col_lon  = find_col(cols, ["경도", "longitude", "lon", "lng"])
    col_place= find_col(cols, ["위치", "지역", "장소", "place", "location"])

    out = pd.DataFrame()
    if col_time:         out["time"]      = pd.to_datetime(df[col_time], errors="coerce")
    if col_mag:          out["magnitude"] = to_num(df[col_mag])
    if col_dep:          out["depth_km"]  = to_num(df[col_dep])
    if col_lat:          out["latitude"]  = parse_coord(df[col_lat], "lat")
    if col_lon:          out["longitude"] = parse_coord(df[col_lon], "lon")
    if col_place:        out["place"]     = df[col_place].astype(str)

    # 범위 필터 & 정렬
    if {"latitude","longitude"}.issubset(out.columns):
        out = out[(out["latitude"].between(-90, 90)) & (out["longitude"].between(-180, 180))]
    if "time" in out.columns:
        out = out.sort_values("time").reset_index(drop=True)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# 입력 UI
# ──────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1,1])
with left:
    st.subheader("📁 데이터 업로드")
    up = st.file_uploader("국외지진목록 파일(.xlsx, .xls, .html, .htm, .csv)", type=["xlsx","xls","html","htm","csv"])
with right:
    st.subheader("ℹ️ 안내")
    st.write(
        "• 위·경도에 N/S/E/W가 붙은 값도 자동 변환됩니다.\n"
        "• 규모(M)는 정수 구간(0.0–0.9, 1.0–1.9, …, 9.0–9.9, 10.0)으로 색상이 달라집니다.\n"
        "• 원 크기는 실제 규모(M)에 비례합니다."
    )

if up is None:
    st.info("파일을 업로드해 주세요. (예: 국외지진목록_5개년.xlsx)")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# 로드 & 정제
# ──────────────────────────────────────────────────────────────────────────────
try:
    raw = load_quakes(up.read(), filename=up.name)
except Exception as e:
    st.error("파일을 읽는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

clean = auto_clean(raw)

# 자동 탐지 실패 시, 수동 매핑 폴백 UI
if not {"latitude","longitude"}.issubset(clean.columns):
    st.warning("자동 위도/경도 탐지에 실패했어요. 아래에서 컬럼을 직접 선택해 주세요.")
    cols = list(raw.columns)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        lat_col = st.selectbox("위도", cols, index=None, placeholder="선택")
    with c2:
        lon_col = st.selectbox("경도", cols, index=None, placeholder="선택")
    with c3:
        mag_col = st.selectbox("규모(선택)", [None]+cols, index=0)
    with c4:
        dep_col = st.selectbox("깊이(선택)", [None]+cols, index=0)
    with c5:
        time_col = st.selectbox("시간(선택)", [None]+cols, index=0)

    clean = pd.DataFrame()
    if time_col: clean["time"] = pd.to_datetime(raw[time_col], errors="coerce")
    if mag_col:  clean["magnitude"] = to_num(raw[mag_col])
    if dep_col:  clean["depth_km"] = to_num(raw[dep_col])
    if lat_col:  clean["latitude"] = parse_coord(raw[lat_col], "lat")
    if lon_col:  clean["longitude"] = parse_coord(raw[lon_col], "lon")
    if {"latitude","longitude"}.issubset(clean.columns):
        clean = clean[(clean["latitude"].between(-90,90)) & (clean["longitude"].between(-180,180))]
        if "time" in clean: clean = clean.sort_values("time").reset_index(drop=True)
    else:
        st.error("위도/경도는 반드시 지정해야 합니다.")
        st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# 필터(선택): 기간/규모/깊이/검색어
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🧭 필터")
    if "time" in clean.columns and clean["time"].notna().any():
        tmin, tmax = pd.to_datetime(clean["time"].min()), pd.to_datetime(clean["time"].max())
        date_range = st.date_input("기간", value=(tmin.date(), tmax.date()), min_value=tmin.date(), max_value=tmax.date())
    else:
        date_range = None

    if "magnitude" in clean.columns and clean["magnitude"].notna().any():
        mag_min, mag_max = float(np.nanmin(clean["magnitude"])), float(np.nanmax(clean["magnitude"]))
        m_lo, m_hi = st.slider("규모(M)", min_value=float(np.floor(mag_min)),
                               max_value=float(np.ceil(mag_max)),
                               value=(float(np.floor(mag_min)), float(np.ceil(mag_max))),
                               step=0.1)
    else:
        m_lo, m_hi = None, None

    if "depth_km" in clean.columns and clean["depth_km"].notna().any():
        dmin, dmax = float(np.nanmin(clean["depth_km"])), float(np.nanmax(clean["depth_km"]))
        dep_lo, dep_hi = st.slider("깊이(km)", min_value=float(max(0.0, np.floor(dmin))),
                                   max_value=float(np.ceil(dmax)),
                                   value=(float(max(0.0, np.floor(dmin))), float(np.ceil(dmax))),
                                   step=1.0)
    else:
        dep_lo, dep_hi = None, None

    place_query = st.text_input("지역/위치 키워드 🔎", value="").strip()

f = clean.copy()
if date_range and "time" in f.columns and f["time"].notna().any():
    start_dt = pd.to_datetime(pd.Timestamp(date_range[0]))
    end_dt   = pd.to_datetime(pd.Timestamp(date_range[1])) + pd.Timedelta(days=1)
    f = f[(f["time"] >= start_dt) & (f["time"] < end_dt)]
if m_lo is not None and m_hi is not None and "magnitude" in f.columns:
    f = f[f["magnitude"].between(m_lo, m_hi)]
if dep_lo is not None and dep_hi is not None and "depth_km" in f.columns:
    f = f[f["depth_km"].between(dep_lo, dep_hi)]
if place_query and "place" in f.columns:
    f = f[f["place"].str.contains(place_query, case=False, na=False)]

# ──────────────────────────────────────────────────────────────────────────────
# 규모 정수 구간 라벨링 & 이산 색상 맵
# ──────────────────────────────────────────────────────────────────────────────
if "magnitude" in f.columns and f["magnitude"].notna().any():
    f["mag_bin"] = make_mag_bin_label(f["magnitude"])
    # 현재 데이터에 실제로 존재하는 구간만 (0→10 순서)
    order_all = [f"{i}.0–{i}.9" for i in range(0,10)] + ["10.0"]
    labels_order = [lab for lab in order_all if lab in set(f["mag_bin"].dropna().unique())]
    color_map = build_color_map(labels_order)
else:
    f["mag_bin"] = np.nan
    labels_order, color_map = [], {}

# ──────────────────────────────────────────────────────────────────────────────
# KPI
# ──────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("표시 건수", f"{len(f):,}")
if "magnitude" in f.columns and f["magnitude"].notna().any():
    k2.metric("평균 규모", f"{f['magnitude'].mean():.2f}")
    k3.metric("최대 규모", f"{f['magnitude'].max():.1f}")
else:
    k2.metric("평균 규모", "-"); k3.metric("최대 규모", "-")
if "depth_km" in f.columns and f["depth_km"].notna().any():
    k4.metric("평균 깊이(km)", f"{f['depth_km'].mean():.0f}")
else:
    k4.metric("평균 깊이(km)", "-")

# ──────────────────────────────────────────────────────────────────────────────
# 지도 (Plotly)
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("🗺️ 정수 구간별 색상 포인트 지도")
hover_cols = [c for c in ["time","magnitude","depth_km","place"] if c in f.columns]
size_col = "magnitude" if "magnitude" in f.columns else None

fig = px.scatter_geo(
    f,
    lat="latitude", lon="longitude",
    size=size_col, size_max=16, opacity=0.8,
    color="mag_bin",
    color_discrete_map=color_map,
    category_orders={"mag_bin": labels_order},
    hover_data=hover_cols,
    projection="natural earth",
)
fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    legend_title_text="규모 구간(M)"
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("📄 데이터 미리보기 (필터 적용 후)"):
    st.dataframe(f.head(100), use_container_width=True)

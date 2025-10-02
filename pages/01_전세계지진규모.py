import io
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────
# 기본 설정
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="🌍 전세계 지진 규모 지도", page_icon="🌍", layout="wide")
st.title("🌍 전세계 지진 규모 지도")
st.caption("정수 규모 구간과 진원 깊이 구간을 한 지도에서 확인할 수 있습니다. (작을수록 노랑→주황→진한 붉은색)")

# ─────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────
def parse_coord(series: pd.Series, kind: str) -> pd.Series:
    """'24.72 N' / '66.67 W' 같은 표기를 숫자로 변환. kind='lat'는 S 음수, 'lon'은 W 음수."""
    s = series.astype(str).str.strip()
    num = pd.to_numeric(s.str.extract(r'([-+]?\d+(?:\.\d+)?)')[0], errors="coerce")
    if kind == "lat":
        return num.mask(s.str.contains(r"[Ss]"), -num)
    else:
        return num.mask(s.str.contains(r"[Ww]"), -num)

def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")

def make_mag_bin_label(m: pd.Series) -> pd.Series:
    """규모를 1.0 단위 구간으로 라벨링(0.0–0.9 … 9.0–9.9, 10.0). 음수/NaN은 NaN."""
    mf = np.floor(m).astype("Int64")
    mf = mf.clip(lower=0, upper=10)
    def lab(v):
        if pd.isna(v): return np.nan
        v = int(v)
        return f"{v}.0–{v}.9" if v < 10 else "10.0"
    return mf.map(lab)

# 사용자 지정 규모 색상 스케일(노랑 → 주황 → 아주 진한 붉은색)
# positions 0.0, 0.55, 1.0 정도로 배치해서 작은 규모는 연하고, 큰 규모는 진하게 보이게 함
MAG_COLORSCALE = [
    (0.00, "#FFF59D"),  # 밝은 노랑 (작은 규모)
    (0.55, "#FB8C00"),  # 선명한 주황 (중간 규모)
    (1.00, "#B71C1C"),  # 아주 진한 붉은색 (큰 규모)
]

def build_mag_colors(labels_order):
    """정수 구간 라벨들을 0→1 구간에 균등 매핑해 MAG_COLORSCALE에서 색 샘플링."""
    if not labels_order:
        return {}
    # 0~10 전체 구간 기준 위치를 계산 (데이터에 존재하는 구간만 뽑음)
    # 0~9: 'x.0–x.9', 10: '10.0' (마지막은 최댓값에 대응)
    def bin_index(label):
        return 10 if label == "10.0" else int(label.split(".")[0])
    idxs = [bin_index(lab) for lab in labels_order]
    pos = np.array(idxs) / 10.0
    # colorscale 샘플
    colors = []
    for p in pos:
        colors.append(px.colors.sample_colorscale(MAG_COLORSCALE, [p])[0])
    return {lab: colors[i] for i, lab in enumerate(labels_order)}

def depth_category(d: pd.Series) -> pd.Series:
    """깊이 구간 라벨: 천발(0–70), 중발(70–300), 심발(>300)"""
    cat = pd.Series(index=d.index, dtype=object)
    cat[(d >= 0) & (d < 70)]  = "천발(0–70km)"
    cat[(d >= 70) & (d <= 300)] = "중발(70–300km)"
    cat[(d > 300)]            = "심발(>300km)"
    return cat

# ─────────────────────────────────────────────────────────
# 로더: 다양한 포맷 자동 판별
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_quakes(file_bytes: bytes, filename: str = "uploaded") -> pd.DataFrame:
    b = file_bytes or b""
    head = b[:64].lstrip()

    # 1) XLSX
    if head.startswith(b"PK"):
        return pd.read_excel(io.BytesIO(b), engine="openpyxl")

    # 2) 구형 XLS
    if head.startswith(b"\xD0\xCF\x11\xE0"):
        return pd.read_excel(io.BytesIO(b), engine="xlrd")

    # 3) HTML (확장자만 xls여도 HTML 테이블인 경우)
    looks_html = head.startswith(b"<!DOCTYPE") or head.startswith(b"<html") or (b"<table" in b[:8192].lower())
    if looks_html:
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
        # BeautifulSoup fallback
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
        raise RuntimeError("HTML 테이블을 찾지 못했습니다. (html5lib/lxml/bs4 모두 실패)")

    # 4) CSV (, / \t / ;)
    for kwargs in [dict(), dict(sep="\t"), dict(sep=";")]:
        try:
            return pd.read_csv(io.BytesIO(b), **kwargs)
        except Exception:
            pass

    raise RuntimeError(f"알 수 없는 파일 형식입니다: {filename}")

# ─────────────────────────────────────────────────────────
# 정제: 자동 감지 (+ 수동 매핑 폴백은 필요 시 제공)
# ─────────────────────────────────────────────────────────
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

    if {"latitude","longitude"}.issubset(out.columns):
        out = out[(out["latitude"].between(-90, 90)) & (out["longitude"].between(-180, 180))]
    if "time" in out.columns:
        out = out.sort_values("time").reset_index(drop=True)
    return out

# ─────────────────────────────────────────────────────────
# 입력 UI
# ─────────────────────────────────────────────────────────
left, right = st.columns([1,1])
with left:
    st.subheader("📁 데이터 업로드")
    up = st.file_uploader("국외지진목록 파일(.xlsx, .xls, .html, .htm, .csv)", type=["xlsx","xls","html","htm","csv"])
with right:
    st.subheader("🧪 표시 모드")
    show_mag   = st.toggle("규모 확인 (원, 구간별-그라데이션 색)", value=True)
    show_depth = st.toggle("깊이 확인 (삼각형, 천·중·심발 색)", value=False)

if up is None:
    st.info("파일을 업로드해 주세요. (예: 국외지진목록_5개년.xlsx)")
    st.stop()

# ─────────────────────────────────────────────────────────
# 로드 & 정제 (+ 필요시 수동 매핑)
# ─────────────────────────────────────────────────────────
try:
    raw = load_quakes(up.read(), filename=up.name)
except Exception as e:
    st.error("파일을 읽는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

clean = auto_clean(raw)
if not {"latitude","longitude"}.issubset(clean.columns):
    st.warning("자동 위도/경도 탐지에 실패했습니다. 아래에서 컬럼을 직접 선택해 주세요.")
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

# ─────────────────────────────────────────────────────────
# 사이드바 필터
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🧭 필터")
    if "time" in clean.columns and clean["time"].notna().any():
        tmin, tmax = pd.to_datetime(clean["time"].min()), pd.to_datetime(clean["time"].max())
        date_range = st.date_input("기간", value=(tmin.date(), tmax.date()),
                                   min_value=tmin.date(), max_value=tmax.date())
    else:
        date_range = None
    if "magnitude" in clean.columns and clean["magnitude"].notna().any():
        mag_min, mag_max = float(np.nanmin(clean["magnitude"])), float(np.nanmax(clean["magnitude"]))
        m_lo, m_hi = st.slider("규모(M)", min_value=float(np.floor(mag_min)),
                               max_value=float(np.ceil(mag_max)),
                               value=(float(np.floor(mag_min)), float(np.ceil(mag_max))), step=0.1)
    else:
        m_lo, m_hi = None, None
    if "depth_km" in clean.columns and clean["depth_km"].notna().any():
        dmin, dmax = float(np.nanmin(clean["depth_km"])), float(np.nanmax(clean["depth_km"]))
        dep_lo, dep_hi = st.slider("깊이(km)", min_value=float(max(0.0, np.floor(dmin))),
                                   max_value=float(np.ceil(dmax)),
                                   value=(float(max(0.0, np.floor(dmin))), float(np.ceil(dmax))), step=1.0)
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

# ─────────────────────────────────────────────────────────
# 레이어별 데이터 준비
# ─────────────────────────────────────────────────────────
traces = []

# (A) 규모 레이어 (원, 구간별-그라데이션 색)
if show_mag and "magnitude" in f.columns and f["magnitude"].notna().any():
    f = f.copy()
    f["mag_bin"] = make_mag_bin_label(f["magnitude"])
    order_all = [f"{i}.0–{i}.9" for i in range(0,10)] + ["10.0"]
    labels_order = [lab for lab in order_all if lab in set(f["mag_bin"].dropna().unique())]
    mag_color_map = build_mag_colors(labels_order)

    # 구간별 trace (범례 가독성)
    for lab in labels_order:
        dfb = f[f["mag_bin"] == lab]
        if dfb.empty: 
            continue
        # 규모가 클수록 원도 조금 더 크게(시각 강조)
        size_vals = np.clip((dfb["magnitude"].fillna(dfb["magnitude"].median())*2.0), 5, 20)
        traces.append(go.Scattergeo(
            lon=dfb["longitude"], lat=dfb["latitude"],
            mode="markers",
            name=f"규모 {lab}",
            legendgroup="magnitude", showlegend=True,
            marker=dict(
                symbol="circle",
                size=size_vals,
                color=mag_color_map[lab],
                line=dict(width=0.5, color="white"),
                opacity=0.9,
            ),
            hovertemplate="<b>규모(M)</b>: %{customdata[0]:.1f}<br>"
                          "위도: %{lat:.2f}, 경도: %{lon:.2f}<br>"
                          "%{customdata[1]}",
            customdata=np.stack([
                dfb["magnitude"].values if "magnitude" in dfb else np.full(len(dfb), np.nan),
                dfb["place"].values if "place" in dfb else np.array([""]*len(dfb))
            ], axis=1)
        ))

# (B) 깊이 레이어 (삼각형)
if show_depth and "depth_km" in f.columns and f["depth_km"].notna().any():
    f = f.copy()
    f["depth_cat"] = depth_category(f["depth_km"])
    depth_order = ["천발(0–70km)", "중발(70–300km)", "심발(>300km)"]
    # 색상: 하늘색, 파란색, 어두운 푸른색
    depth_colors = {
        "천발(0–70km)": "#87CEEB",  # SkyBlue
        "중발(70–300km)": "#1976D2", # Blue-ish
        "심발(>300km)": "#0D47A1",  # Dark Blue
    }
    for lab in depth_order:
        dfd = f[f["depth_cat"] == lab]
        if dfd.empty:
            continue
        traces.append(go.Scattergeo(
            lon=dfd["longitude"], lat=dfd["latitude"],
            mode="markers",
            name=f"깊이 {lab}",
            legendgroup="depth", showlegend=True,
            marker=dict(
                symbol="triangle-up",
                size=11,
                color=depth_colors[lab],
                line=dict(width=0.5, color="white"),
                opacity=0.95,
            ),
            hovertemplate="<b>깊이</b>: %{customdata[0]} km<br>"
                          "위도: %{lat:.2f}, 경도: %{lon:.2f}<br>"
                          "%{customdata[1]}",
            customdata=np.stack([
                dfd["depth_km"].round(0).astype("Int64").astype(str).replace("<NA>","-").values
                    if "depth_km" in dfd else np.array(["-"]*len(dfd)),
                dfd["place"].values if "place" in dfd else np.array([""]*len(dfd))
            ], axis=1)
        ))

# ─────────────────────────────────────────────────────────
# 지도 렌더링
# ─────────────────────────────────────────────────────────
if not traces:
    st.info("오른쪽 상단의 토글(규모 확인 / 깊이 확인)을 켜고 지도를 확인하세요.")
else:
    fig = go.Figure(data=traces)
    fig.update_layout(
        geo=dict(
            projection=dict(type="natural earth"),
            showcountries=True,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(title="레이어", orientation="h", yanchor="bottom", y=0.02, xanchor="left", x=0.02)
    )
    st.plotly_chart(fig, use_container_width=True)

# 데이터 미리보기
with st.expander("📄 데이터 미리보기 (필터 적용 후)"):
    show_cols = [c for c in ["time","magnitude","depth_km","place","latitude","longitude"] if c in f.columns]
    st.dataframe(f[show_cols].head(100), use_container_width=True)

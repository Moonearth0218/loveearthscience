# app.py
import io
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="🌏 지진 깊이 분류 지도", layout="wide")

st.title("🌏 지진 깊이 분류 지도 (천·중·심발) with Emoji")

st.markdown("""
학생이 **CSV 파일**만 업로드하면, **진원 깊이(Depth)**에 따라  
- 🟢 **천발(Shallow)**,  
- 🟠 **중발(Intermediate)**,  
- 🔵 **심발(Deep)**  
로 자동 분류하여 **지도로 시각화**하고, **분류 결과 CSV**를 내려받을 수 있습니다.
""")

# --- 사이드바: 설정 ---
with st.sidebar:
    st.header("⚙️ 분류 기준 설정")
    shallow_max = st.number_input("천발 상한 (km, 미만)", value=70, min_value=0, max_value=3000, step=1)
    intermediate_max = st.number_input("중발 상한 (km, 미만)", value=300, min_value=0, max_value=3000, step=1,
                                       help="중발 구간은 [천발 상한, 중발 상한) 입니다. 그 이상은 심발로 분류.")
    st.caption("일반적으로 천발 < 70 km, 70–300 km 중발, ≥300 km 심발로 많이 사용합니다.")

    st.header("📄 CSV 업로드")
    f = st.file_uploader("지진 CSV 업로드 (UTF-8 권장)", type=["csv"])

    st.divider()
    st.subheader("🧩 컬럼 자동 매핑 도움말")
    st.caption("위도/경도/깊이 컬럼명을 자동 탐지합니다. 필요하면 아래에서 직접 지정하세요.")

# --- 샘플/템플릿 제공 ---
sample = pd.DataFrame({
    "latitude": [35.2, -6.7, -17.4, 38.1, 28.5],
    "longitude": [141.1, 155.1, -70.2, -122.3, -178.2],
    "depth_km": [10, 85, 410, 260, 540],
    "magnitude": [6.9, 5.4, 6.2, 4.8, 6.0],
    "time": ["2024-11-22", "2024-07-03", "2023-05-10", "2022-09-01", "2021-01-15"]
})
template = pd.DataFrame({
    "latitude": [], "longitude": [], "depth_km": [], "magnitude": [], "time": []
})

c1, c2 = st.columns(2)
with c1:
    st.markdown("**예시 데이터 미리보기**")
    st.dataframe(sample, use_container_width=True, height=180)
    st.download_button("⬇️ 예시 CSV 다운로드", data=sample.to_csv(index=False).encode("utf-8"),
                       file_name="earthquakes_sample.csv", mime="text/csv")
with c2:
    st.markdown("**업로드용 템플릿**")
    st.dataframe(template, use_container_width=True, height=180)
    st.download_button("⬇️ 템플릿 CSV 다운로드", data=template.to_csv(index=False).encode("utf-8"),
                       file_name="earthquakes_template.csv", mime="text/csv")

st.divider()

# --- 데이터 로딩 ---
def auto_map_columns(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    # 후보 키워드
    lat_key = next((cols[k] for k in cols if k in ["lat", "latitude", "y", "위도"]), None)
    lon_key = next((cols[k] for k in cols if k in ["lon", "longitude", "x", "경도"]), None)
    depth_key = next((cols[k] for k in cols if k in ["depth", "depth_km", "z", "깊이", "depth(km)"]), None)
    return lat_key, lon_key, depth_key

def classify_depth(depth, shallow_max=70, intermediate_max=300):
    if pd.isna(depth):
        return "알수없음"
    if depth < shallow_max:
        return "천발"
    if depth < intermediate_max:
        return "중발"
    return "심발"

def emoji_for(cat):
    return {"천발":"🟢","중발":"🟠","심발":"🔵"}.get(cat, "❔")

def color_for(cat):
    # RGB
    return {
        "천발": [34, 197, 94],     # green-ish
        "중발": [234, 179, 8],     # amber-ish
        "심발": [59, 130, 246],    # blue-ish
        "알수없음": [148, 163, 184] # gray
    }.get(cat, [148,163,184])

df = None
if f:
    try:
        df = pd.read_csv(f)
    except Exception:
        f.seek(0)
        df = pd.read_csv(f, encoding="cp949")  # 한글 CSV 대응

if df is None:
    st.info("왼쪽에서 CSV를 업로드하면 자동 분류와 지도가 생성됩니다. (위 예시 CSV로 테스트 가능)")
    st.stop()

# --- 컬럼 매핑 UI ---
lat_key, lon_key, depth_key = auto_map_columns(df)

st.subheader("📌 컬럼 매핑")
mc1, mc2, mc3, mc4 = st.columns([1,1,1,2])
with mc1:
    lat_key = st.selectbox("위도(Column)", options=df.columns.tolist(), index=(df.columns.tolist().index(lat_key) if lat_key in df.columns else 0))
with mc2:
    lon_key = st.selectbox("경도(Column)", options=df.columns.tolist(), index=(df.columns.tolist().index(lon_key) if lon_key in df.columns else 1 if len(df.columns)>1 else 0))
with mc3:
    depth_key = st.selectbox("깊이(km, Column)", options=df.columns.tolist(), index=(df.columns.tolist().index(depth_key) if depth_key in df.columns else 2 if len(df.columns)>2 else 0))
with mc4:
    mag_key = st.selectbox("규모(선택)", options=["<없음>"] + df.columns.tolist(), index=0)

# --- 전처리 & 분류 ---
work = df.copy()
# 숫자 변환 시도
for c in [lat_key, lon_key, depth_key]:
    work[c] = pd.to_numeric(work[c], errors="coerce")

work["분류"] = work[depth_key].apply(lambda d: classify_depth(d, shallow_max, intermediate_max))
work["emoji"] = work["분류"].apply(emoji_for)
work["color"] = work["분류"].apply(color_for)
if mag_key != "<없음>":
    work["magnitude"] = pd.to_numeric(work[mag_key], errors="coerce")
else:
    work["magnitude"] = np.nan

# 중심 위치 계산
valid = work[[lat_key, lon_key]].dropna()
center_lat = float(valid[lat_key].mean()) if len(valid) else 0.0
center_lon = float(valid[lon_key].mean()) if len(valid) else 0.0

# --- 요약 카드 ---
st.subheader("📈 분류 요약")
k1,k2,k3,k4 = st.columns(4)
k1.metric("🟢 천발", int((work["분류"]=="천발").sum()))
k2.metric("🟠 중발", int((work["분류"]=="중발").sum()))
k3.metric("🔵 심발", int((work["분류"]=="심발").sum()))
k4.metric("❔ 알수없음", int((work["분류"]=="알수없음").sum()))

# --- 데이터 미리보기 ---
with st.expander("🔎 분류 결과 표 보기"):
    show_cols = [lat_key, lon_key, depth_key, "분류", "emoji"]
    if mag_key != "<없음>":
        show_cols.append("magnitude")
    st.dataframe(work[show_cols], use_container_width=True, height=280)

# --- 지도 시각화 (두 가지 레이어: 색 점 + 이모지 텍스트) ---
st.subheader("🗺️ 지도 시각화")

# ScatterplotLayer: 색상으로 구분
scatter = pdk.Layer(
    "ScatterplotLayer",
    data=work,
    get_position=[lon_key, lat_key],
    get_radius= "np.clip((magnitude*10000) if magnitude==magnitude else 5000, 3000, 25000)",  # NaN 방지
    radius_min_pixels=4,
    radius_max_pixels=30,
    get_fill_color="color",
    pickable=True
)

# TextLayer: 이모지로 구분
text = pdk.Layer(
    "TextLayer",
    data=work,
    get_position=[lon_key, lat_key],
    get_text="emoji",
    get_size=16,  # 픽셀 크기
    get_alignment_baseline="'bottom'",
    get_color=[0,0,0,240],  # 글자 테두리 효과는 제한적이므로 검정
    pickable=True
)

tooltip = {
    "html": """
    <div style="font-size:14px">
      <b>{emoji} {분류}</b><br/>
      위도: {""" + lat_key + """}<br/>
      경도: {""" + lon_key + """}<br/>
      깊이(km): {""" + depth_key + """}<br/>
      규모: {magnitude}
    </div>
    """,
    "style": {"backgroundColor": "white", "color": "black"}
}

deck = pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=2),
    layers=[scatter, text],
    tooltip=tooltip
)

st.pydeck_chart(deck, use_container_width=True)

# --- 범례 ---
st.markdown("""
**범례**  
🟢 천발(Shallow) · 🟠 중발(Intermediate) · 🔵 심발(Deep) · ❔ 알수없음  
점의 **색**과 **이모지**가 동일한 분류를 의미합니다.  
(점 크기는 규모 컬럼이 있는 경우 대략 반영됩니다.)
""")

# --- 히스토그램(깊이 분포) ---
with st.expander("📊 깊이 히스토그램 보기"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    depths = work[depth_key].dropna()
    ax.hist(depths, bins=30)
    ax.set_xlabel("깊이 (km)")
    ax.set_ylabel("개수")
    ax.set_title("지진 깊이 분포")
    st.pyplot(fig, use_container_width=True)

# --- 결과 다운로드 ---
out_cols = df.columns.tolist()
out = work.copy()
out["depth_category"] = work["분류"]
out["depth_emoji"] = work["emoji"]
bytes_out = out.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ 분류 결과 CSV 다운로드", data=bytes_out, file_name="earthquakes_classified.csv", mime="text/csv")

st.caption("© 지구과학 수업용 예제. 지도 배경: Mapbox(스트림릿 기본 키). 업로드 데이터는 세션 메모리에만 머무르며 서버에 저장하지 않습니다.")

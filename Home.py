# Home.py
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="🌍 지진 데이터 랩", layout="wide")
st.title("🌍 지진 데이터 랩")
st.markdown("""
**GitHub Pages/Streamlit Cloud**로 배포 가능한 멀티 페이지 앱입니다.  
왼쪽 **Pages**에서:
- `00_World_Quakes` : 전세계 지진을 **규모별 색**으로 보고, **나라를 선택**하면 해당 국가의 **최신 3건**을 보여줍니다.
- `01_Korea_Depth_Quakes` : 국내 지진을 **깊이별(천·중·심)** **이모지/색**으로 표시하고, **핵사곤 밀도**로 어느 지역에 자주 일어나는지 한눈에 봅니다.
""")

st.subheader("📦 샘플 데이터")
sample_global = pd.DataFrame({
    "latitude":[38.322,-6.204,-17.4,35.2,28.5],
    "longitude":[142.369,155.9,-70.2,141.1,-178.2],
    "magnitude":[9.1,7.5,6.2,6.9,6.0],
    "time":["2011-03-11T05:46:24Z","2024-02-09T12:30:00Z","2023-05-10T03:12:00Z","2024-11-22T11:00:00Z","2021-01-15T00:00:00Z"]
})
sample_kor = pd.DataFrame({
    "latitude":[36.1, 35.7, 37.8, 34.9, 36.4],
    "longitude":[129.4, 129.2, 128.6, 126.7, 127.9],
    "depth_km":[12, 85, 260, 8, 340],
    "magnitude":[3.1, 4.0, 4.6, 2.8, 5.0],
    "time":["2025-04-01T03:12:00Z","2025-06-02T14:05:00Z","2024-12-21T20:40:00Z","2025-02-12T09:10:00Z","2023-10-03T01:00:00Z"]
})

c1,c2 = st.columns(2)
with c1:
    st.markdown("**sample_global_quakes.csv**")
    st.dataframe(sample_global, use_container_width=True, height=200)
    st.download_button("⬇️ 다운로드", sample_global.to_csv(index=False).encode("utf-8"),
                       "sample_global_quakes.csv", "text/csv")
with c2:
    st.markdown("**sample_korea_quakes.csv**")
    st.dataframe(sample_kor, use_container_width=True, height=200)
    st.download_button("⬇️ 다운로드", sample_kor.to_csv(index=False).encode("utf-8"),
                       "sample_korea_quakes.csv", "text/csv")

st.caption("※ 업로드한 데이터는 서버에 저장하지 않으며 세션 내에서만 사용됩니다.")

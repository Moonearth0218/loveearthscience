# Home.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="🌍 지진 데이터 랩", layout="wide")
st.title("🌍 지진 데이터 랩")

st.markdown("""
왼쪽 **Pages**에서:
- **00_전세계지진규모**: 전세계 지진을 **규모별 색**으로 보고, **나라 선택** 시 그 나라의 **최신 3건**을 표로 확인합니다.
- **01_국내_심중천분류**: 국내 지진을 **깊이(천·중·심)**로 **이모지/색** 표시 + **핵사곤 밀도**로 자주 일어나는 지역을 살펴봅니다.
""")

st.subheader("📦 샘플 안내")
st.write("- `data/국외지진목록_2015-01-01_2025-09-29.xls` 파일을 자동으로 사용(00페이지).")
st.write("- 01페이지는 국내 CSV 업로드 또는 국내만 필터링한 CSV를 사용하면 좋아요.")

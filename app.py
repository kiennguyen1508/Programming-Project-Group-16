import streamlit as st
from panels import Charlie, Kien, Matthew, Michael

st.set_page_config(page_title="Econ Dashboard", layout="wide")
st.title("Economics & Finance Dashboard")

top = st.columns(2)
bottom = st.columns(2)

with top[0]:
    Charlie.render()
with top[1]:
    Kien.render()
with bottom[0]:
    Matthew.render()
with bottom[1]:
    Michael.render()
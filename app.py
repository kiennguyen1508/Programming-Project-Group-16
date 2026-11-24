import streamlit as st
from panels import Charlie, Kien, Matthew, Michael

st.set_page_config(page_title="What Drives Gold? An Interactive Analysis of Returns, Correlations, Risk and Market Links: 2015-2026", layout="wide")

# 1. One entry per panel: its title, the quesion it answers,
#    and the function that draws the full panel
PANELS = {
    "regression" : {
        "title" : "Regression Analysis: Silver, Oil and Gold",
        "question" : "How much does silver move with gold, and did that change through its boom and crash?",
        "full" : Charlie.render,
        "preview": getattr(Charlie, "overview", None),
    },
    "performance" : {
        "title" : "Performance and Ratios",
        "question" : "How has gold performed, and does it protect investors when markets are stressed?",
        "full" : Kien.render,
        "preview": getattr(Kien, "overview", None),
    },
    "correlation" : {
        "title" : "Correlation Heatmap",
        "question" : "Which assets move together, and has that changed?",
        "full" : Matthew.render,
        "preview": getattr(Matthew, "overview", None),
    },
    "risk" : {
        "title" : "Distribution and Risk",
        "question" : "How bad is a bad day for gold?",
        "full": getattr(Michael, "render", lambda: st.info("Michael's panel is not ready yet.")),
        "preview": getattr(Michael, "overview", None),
    }
}

# 2. Remember which panel is open. None means "show the overview"
if "focus" not in st.session_state:
    st.session_state.focus = None

def open_panel(key):
    st.session_state.focus = key

# 3. What one overview card looks like (placeholder for now)
def overview_card(key, info):
    number = list(PANELS).index(key) + 1
    with st.container(border=True, height=560):
        left, right = st.columns([3, 1], vertical_alignment="top")
        with left:
            st.caption(f"PANEL {number:02d}")
            st.subheader(info["title"], anchor=False)
            st.caption(info["question"])
        with right:
            st.button("Open →", key=f"open_{key}", width="stretch",
                      on_click=open_panel, args=(key,))
        preview = info.get("preview")
        if preview:
            preview()
        else:
            st.markdown(
                "<div style='height:280px; border:1px dashed #2B3140; "
                "border-radius:12px; display:flex; align-items:center; "
                "justify-content:center; color:#8A91A3; font-size:14px'>"
                "Preview coming soon</div>",
                unsafe_allow_html=True,
            )


# 4. Draw the page
if st.session_state.focus is None:
    st.caption("PROGRAMMING PROJECT · GROUP 16")
    st.title("What Drives Gold? An Interactive Analysis of Returns, Correlations, Risk and Market Links", anchor=False)
    st.caption("Daily data from 2015 to Oct 2026. Open any panel to explore it in full.")
    keys = list(PANELS)
    for row in (keys[:2], keys[2:]):          # two rows of two
        cols = st.columns(2)
        for col, key in zip(cols, row):
            with col:
                overview_card(key, PANELS[key])
else:
    info = PANELS[st.session_state.focus]
    st.button("← Back to overview", on_click=open_panel, args=(None,))
    info["full"]()  


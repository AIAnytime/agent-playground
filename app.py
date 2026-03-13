"""
LangGraph Playground -- Interactive exploration of LangGraph, agent design,
memory systems, and context engineering.

Run: streamlit run playground/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="LangGraph Playground",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS for a professional look --
st.markdown("""
<style>
    /* Clean, professional styling */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #dee2e6;
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        font-size: 1.4rem;
        color: #212529;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid #dee2e6;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.25rem;
        font-weight: 500;
        color: #495057;
        border-bottom: 2px solid transparent;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #212529;
        border-bottom-color: #0d6efd;
        background-color: transparent;
    }
    /* Headers */
    h1, h2, h3 {
        color: #212529;
    }
    /* Info/Warning boxes */
    .stAlert {
        border-radius: 4px;
    }
    /* Text areas */
    .stTextArea textarea {
        font-family: "SF Mono", "Fira Code", monospace;
        font-size: 0.85rem;
    }
    /* Buttons */
    .stButton > button {
        border-radius: 4px;
        font-weight: 500;
    }
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #212529;
    }
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils import init_session_defaults, render_sidebar
from tabs import tab_langgraph_basics
from tabs import tab_agent_design
from tabs import tab_scratchpad
from tabs import tab_checkpointing
from tabs import tab_memory_systems
from tabs import tab_context_engineering

# Initialize session state
init_session_defaults()

# Sidebar
render_sidebar()

# Main content -- tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "LangGraph Basics",
    "Agent Design",
    "Scratchpad",
    "Checkpointing",
    "Memory Systems",
    "Context Engineering",
])

with tab1:
    tab_langgraph_basics.render()

with tab2:
    tab_agent_design.render()

with tab3:
    tab_scratchpad.render()

with tab4:
    tab_checkpointing.render()

with tab5:
    tab_memory_systems.render()

with tab6:
    tab_context_engineering.render()

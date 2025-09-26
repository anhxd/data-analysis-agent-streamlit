import os, shutil
import streamlit as st
from dotenv import load_dotenv

from tools.csv_tool import CSVTool
from tools.sql_tool import SQLTool
from tools.plot_tool import PlotTool
from tools.stats_tool import StatsTool
from main import run_agent, ToolSuite

load_dotenv()

APP_HOST = os.getenv("APP_HOST","0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT","7860"))  # not used by Streamlit Cloud
UPLOAD_DIR = os.getenv("UPLOAD_DIR","uploads")
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR","outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

st.set_page_config(page_title="Data Analysis Agent", layout="wide")
st.title("🔍 Data Analysis Agent (Streamlit)")

with st.sidebar:
    st.subheader("Settings")
    openai_api_key = st.text_input("OPENAI_API_KEY", value=os.getenv("OPENAI_API_KEY",""), type="password")
    openai_model = st.text_input("OPENAI_MODEL", value=os.getenv("OPENAI_MODEL","gpt-4o-mini"))
    openai_temp = st.number_input("OPENAI_TEMPERATURE", min_value=0.0, max_value=1.0, value=float(os.getenv("OPENAI_TEMPERATURE","0.2")), step=0.05)
    st.caption("Keys are kept in memory only; for Streamlit Cloud, set secrets instead.")

st.markdown("Upload a CSV/XLSX or use the bundled sample. Then enter a task and hit **Run Agent**.")

query = st.text_area("Task", "Group revenue by month; find anomalies; plot top-5 categories.")
file = st.file_uploader("Upload CSV/XLSX (optional)", type=["csv","xlsx","xls"])

run = st.button("Run Agent")

if run:
    # ensure OPENAI_API_KEY in env for main.py client
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if openai_model:
        os.environ["OPENAI_MODEL"] = openai_model
    os.environ["OPENAI_TEMPERATURE"] = str(openai_temp)

    if file is None:
        data_path = "data/sample_sales.csv"
    else:
        fname = os.path.basename(file.name)
        dest = os.path.join(UPLOAD_DIR, fname)
        with open(dest, "wb") as f:
            f.write(file.getbuffer())
        data_path = dest

    tools = ToolSuite(CSVTool(), SQLTool(), PlotTool(OUTPUTS_DIR), StatsTool())
    try:
        out = run_agent(query, tools, df_default_path=data_path)
    except Exception as e:
        out = f"ERROR: {e}"

    st.subheader("Agent Output")
    st.code(out)

    # show generated plots
    images = sorted([os.path.join(OUTPUTS_DIR, f) for f in os.listdir(OUTPUTS_DIR) if f.endswith(".png")])
    if images:
        st.subheader("Generated Plots")
        for img in images[-12:]:  # show last few
            st.image(img, use_column_width=True)
    else:
        st.info("No plots generated yet. Ask the agent to create one (e.g., 'plot ...').")

st.divider()
st.caption("Tip: Set your OPENAI_API_KEY in Streamlit secrets. For local dev, set it in .env.")

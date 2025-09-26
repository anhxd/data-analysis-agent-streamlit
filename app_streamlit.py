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
APP_PORT = int(os.getenv("APP_PORT","7860"))  # Not used by Streamlit Cloud, local only
UPLOAD_DIR = os.getenv("UPLOAD_DIR","uploads")
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR","outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

st.set_page_config(page_title="Data Analysis Agent", layout="wide")
st.title("🔍 Data Analysis Agent (Streamlit, HF Inference)")

with st.sidebar:
    st.subheader("Model Settings")
    hf_token = st.text_input("HF_TOKEN", value=os.getenv("HF_TOKEN",""), type="password")
    hf_model = st.text_input("HF_MODEL", value=os.getenv("HF_MODEL","Qwen/Qwen2.5-7B-Instruct"))
    hf_temp = st.number_input("HF_TEMPERATURE", min_value=0.0, max_value=1.0, value=float(os.getenv("HF_TEMPERATURE","0.2")), step=0.05)
    hf_max_new = st.number_input("HF_MAX_NEW_TOKENS", min_value=64, max_value=4096, value=int(float(os.getenv("HF_MAX_NEW_TOKENS","512"))), step=64)
    st.caption("Tip: On Streamlit Cloud, set these in **Secrets**. Locally, use `.env`.")

st.markdown("Upload a CSV/XLSX or use the bundled sample. Enter a task and click **Run Agent**.")

query = st.text_area("Task", "Group revenue by month; find anomalies; plot top-5 categories.")
file = st.file_uploader("Upload CSV/XLSX (optional)", type=["csv","xlsx","xls"])

if st.button("Run Agent"):
    # Ensure env vars for main.py’s HF client
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    if hf_model:
        os.environ["HF_MODEL"] = hf_model
    os.environ["HF_TEMPERATURE"] = str(hf_temp)
    os.environ["HF_MAX_NEW_TOKENS"] = str(hf_max_new)

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

    images = sorted([os.path.join(OUTPUTS_DIR, f) for f in os.listdir(OUTPUTS_DIR) if f.endswith(".png")])
    if images:
        st.subheader("Generated Plots")
        for img in images[-12:]:
            st.image(img, use_column_width=True)
    else:
        st.info("No plots yet. Ask the agent to create one (e.g., 'plot ...').")

st.divider()
st.caption("Uses Hugging Face Inference API. For private deployments, store tokens in Streamlit Secrets.")

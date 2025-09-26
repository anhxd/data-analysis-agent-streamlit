import os, shutil
import gradio as gr
from dotenv import load_dotenv

from tools.csv_tool import CSVTool
from tools.sql_tool import SQLTool
from tools.plot_tool import PlotTool
from tools.stats_tool import StatsTool
from main import run_agent, ToolSuite

load_dotenv()

APP_HOST = os.getenv("APP_HOST","0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT","7860"))
UPLOAD_DIR = os.getenv("UPLOAD_DIR","uploads")
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR","outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def analyze(query, data_file):
    if data_file is None:
        data_path = "data/sample_sales.csv"
    else:
        fname = os.path.basename(data_file.name)
        dest = os.path.join(UPLOAD_DIR, fname)
        shutil.copyfile(data_file.name, dest)
        data_path = dest

    tools = ToolSuite(CSVTool(), SQLTool(), PlotTool(OUTPUTS_DIR), StatsTool())
    try:
        out = run_agent(query, tools, df_default_path=data_path)
    except Exception as e:
        out = f"ERROR: {e}"
    plots = sorted([os.path.join(OUTPUTS_DIR, f) for f in os.listdir(OUTPUTS_DIR) if f.endswith(".png")])
    return out, plots

with gr.Blocks(title="Data Analysis Agent (Gradio)") as demo:
    gr.Markdown("# 🔍 Data Analysis Agent (Gradio — No Docker)")
    with gr.Row():
        query = gr.Textbox(label="Your task", lines=4, value="Group revenue by month; find anomalies; plot top-5 categories.")
        data_file = gr.File(label="Upload CSV/XLSX (optional)", file_types=[".csv",".xlsx",".xls"], file_count="single")
    run_btn = gr.Button("Run Agent")
    out_text = gr.Textbox(label="Agent Output", lines=16)
    gallery = gr.Gallery(label="Generated Plots", columns=2, height=400)
    run_btn.click(analyze, inputs=[query, data_file], outputs=[out_text, gallery])

if __name__ == "__main__":
    demo.queue().launch(server_name=APP_HOST, server_port=APP_PORT)

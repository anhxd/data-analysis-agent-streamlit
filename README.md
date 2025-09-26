# Data Analysis Agent — Streamlit (No Docker)

This is a Gradio-only project (no Docker) for an LLM-driven **Data Analysis Agent**.

- Upload CSV/XLSX (or use `data/sample_sales.csv`)
- The agent plans tool use (CSV/SQL/STATS/PLOT), runs analysis, and outputs a concise summary
- Plots are saved in `outputs/` and previewed in the UI
- Uses `.env` for secrets (included for your private repo)

## Setup

```bash
pip install -r requirements.txt
# Put your tokens in the .env (already included)
python app_streamlit.py
# open http://localhost:7860
```

## Env configuration

Edit `.env`:
```
APP_HOST=0.0.0.0
APP_PORT=7860
UPLOAD_DIR=uploads
OUTPUTS_DIR=outputs

# OpenAI default
OPENAI_API_KEY=sk-REPLACE_ME
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.2

# Optional local provider
# LLM_PROVIDER=local
# LLM_BASE_URL=http://localhost:8000
# LLM_MODEL=llama3.1:8b-instruct
```

## Files
```
app_streamlit.py
main.py
tools/
  csv_tool.py
  sql_tool.py
  plot_tool.py
  stats_tool.py
data/sample_sales.csv
outputs/
.env
requirements.txt
README.md
```


## Deploy to Streamlit Cloud

1) Push this repo to GitHub (private or public).

2) On https://share.streamlit.io, create a new app pointing to `app_streamlit.py`.

3) In **App Settings → Secrets**, add:

```
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = "0.2"
UPLOAD_DIR = "uploads"
OUTPUTS_DIR = "outputs"
```
4) Deploy. Streamlit Cloud installs `requirements.txt` and runs the app.


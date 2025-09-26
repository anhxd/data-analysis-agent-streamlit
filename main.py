import os, json, re
from dataclasses import dataclass
from typing import Dict, Any

from tools.csv_tool import CSVTool
from tools.sql_tool import SQLTool
from tools.plot_tool import PlotTool
from tools.stats_tool import StatsTool

def call_llm(system: str, messages: list[dict], model: str = None, temperature: float = 0.2) -> str:
    provider = os.getenv("LLM_PROVIDER", "openai")
    if provider == "local":
        import requests
        base = os.getenv("LLM_BASE_URL", "http://localhost:8000")
        mdl = os.getenv("LLM_MODEL", "llama3.1:8b-instruct")
        resp = requests.post(f"{base}/chat", json={
            "model": mdl,
            "messages": [{"role":"system","content":system}, *messages],
            "temperature": temperature,
        }, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message","")
    else:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY or configure local provider in .env")
        client = OpenAI(api_key=api_key)
        model = model or os.getenv("OPENAI_MODEL","gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system}, *messages],
            temperature=float(os.getenv("OPENAI_TEMPERATURE", temperature)),
        )
        return resp.choices[0].message.content

SYSTEM = """You are a careful data-analysis agent that plans with a ReAct loop.
When you need a tool, emit EXACTLY one JSON line:
{"tool":"CSV","args":{"cmd":"load","path":"data/sample_sales.csv"}}
Valid tools: CSV, SQL, STATS, PLOT.
After you get TOOL_RESULT, continue planning or finish with a short, clear summary.
List any saved plot paths in the final answer.
"""

@dataclass
class ToolSuite:
    csv: CSVTool
    sql: SQLTool
    plot: PlotTool
    stats: StatsTool

def dispatch_tool(name: str, args: Dict[str,Any], tools: ToolSuite) -> str:
    if name == "CSV":
        cmd = args.get("cmd","load")
        if cmd == "load":
            return tools.csv.load(args["path"])
        if cmd == "head":
            return tools.csv.head(args.get("n",5))
        if cmd == "info":
            return tools.csv.info()
        if cmd == "describe":
            return tools.csv.describe()
        if cmd == "groupby_agg":
            return tools.csv.groupby_agg(args["by"], args["agg_map"])
        if cmd == "filter_query":
            return tools.csv.filter_query(args["expr"])
        raise ValueError(f"Unknown CSV cmd: {cmd}")
    if name == "SQL":
        cmd = args.get("cmd","query")
        if cmd == "load_from_df":
            return tools.sql.load_from_df(tools.csv.get_df(), args.get("table_name","df"))
        if cmd == "query":
            return tools.sql.query(args["sql"], args.get("limit", 50))
        raise ValueError(f"Unknown SQL cmd: {cmd}")
    if name == "STATS":
        cmd = args.get("cmd","corr")
        df = tools.csv.get_df()
        if cmd == "corr":
            return tools.stats.corr(df, args["cols"])
        if cmd == "ttest":
            return tools.stats.ttest(df, args["col"], args["by"])
        if cmd == "z_anomalies":
            return tools.stats.z_anomalies(df[args["col"]], args.get("threshold",3.0))
        raise ValueError(f"Unknown STATS cmd: {cmd}")
    if name == "PLOT":
        cmd = args.get("cmd","line")
        df = tools.csv.get_df()
        if cmd == "line":
            return tools.plot.line(df, args["x"], args["y"], args.get("title",""))
        if cmd == "bar":
            return tools.plot.bar(df, args["x"], args["y"], args.get("title",""))
        if cmd == "hist":
            return tools.plot.hist(df[args["col"]], args.get("bins",20), args.get("title",""))
        raise ValueError(f"Unknown PLOT cmd: {cmd}")
    raise ValueError(f"Unknown tool: {name}")

def run_agent(user_query: str, tools: ToolSuite, df_default_path: str = "data/sample_sales.csv", max_steps: int = 10) -> str:
    messages: list[dict] = [
        {"role":"user","content": f"""{user_query}
If you need data, use CSV.load('{df_default_path}')."""}
    ]
    for _ in range(max_steps):
        reply = call_llm(SYSTEM, messages)
        import re, json
        m = re.search(r"\{\s*\"tool\"\s*:\s*\"(CSV|SQL|STATS|PLOT)\"\s*,\s*\"args\"\s*:\s*(\{.*?\})\s*\}", reply, re.S)
        if m:
            t, args = m.group(1), json.loads(m.group(2))
            try:
                obs = dispatch_tool(t, args, tools)
            except Exception as e:
                obs = f"TOOL-ERROR: {e}"
            messages.append({"role":"assistant","content":reply})
            messages.append({"role":"system","content": f"TOOL_RESULT:\n{obs}"})
            continue
        messages.append({"role":"assistant","content":reply})
        if any(k in reply.lower() for k in ["final answer","here's the summary","summary:"]):
            return reply
    return messages[-1]["content"]

import os, json, re
from dataclasses import dataclass
from typing import Dict, Any, List

from tools.csv_tool import CSVTool
from tools.sql_tool import SQLTool
from tools.plot_tool import PlotTool
from tools.stats_tool import StatsTool

# -------------------- LLM (Hugging Face Inference API) --------------------
# We turn chat-style messages into a single prompt string and call text-generation.
# This works well with instruct models (e.g., Qwen2.5-7B-Instruct, Llama-3 Instruct, Mistral-Instruct).
from huggingface_hub import InferenceClient

def _messages_to_prompt(system: str, messages: List[dict]) -> str:
    """
    Convert chat messages to a generic instruction-style prompt that most
    HF instruct models will follow. Keeps tool JSON detectable.
    """
    buf = []
    if system:
        buf.append("### System\n" + system.strip())
    for m in messages:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if role == "user":
            buf.append("### User\n" + content)
        elif role == "assistant":
            buf.append("### Assistant\n" + content)
        elif role == "system":
            # Use this to inject tool results back to the model
            buf.append("### Tool/Context\n" + content)
        else:
            buf.append(f"### {role.capitalize()}\n" + content)
    # Ask model to continue as Assistant
    buf.append("### Assistant")
    return "\n\n".join(buf).strip()

def call_llm(system: str, messages: List[dict], model: str = None, temperature: float = 0.2) -> str:
    model = model or os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Add it to your .env or Streamlit Secrets.")
    client = InferenceClient(model=model, token=token)

    prompt = _messages_to_prompt(system, messages)
    max_new_tokens = int(float(os.getenv("HF_MAX_NEW_TOKENS", "512")))
    temperature = float(os.getenv("HF_TEMPERATURE", temperature))

    # Some HF models stream tokens; we’ll request a single concatenated string
    text = client.text_generation(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        repetition_penalty=1.05,
        return_full_text=False,
    )
    # The client returns a plain string for text_generation
    return text

# -------------------- Agent System Prompt --------------------
SYSTEM = """\
You are a careful data-analysis agent that plans with a ReAct loop.

When you need a tool, emit EXACTLY one JSON line by itself (no extra text):
{"tool":"CSV","args":{"cmd":"load","path":"data/sample_sales.csv"}}

Valid tools: CSV, SQL, STATS, PLOT.
- CSV: cmds = load(path), head(n), info(), describe(), groupby_agg(by=list, agg_map=dict), filter_query(expr)
- SQL:  cmds = load_from_df(table_name="df"), query(sql, limit=50)
- STATS:cmds = corr(cols=list), ttest(col, by), z_anomalies(col, threshold)
- PLOT: cmds = line(x, y, title?), bar(x, y, title?), hist(col, bins?, title?)

After each tool call you will receive TOOL_RESULT with the output. Use it to decide next step.
Finish with a short, clear summary and list any saved plot paths, if any.
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
    messages: List[dict] = [
        {"role":"user","content": f"""{user_query}
If you need data, use CSV.load('{df_default_path}')."""}
    ]
    for _ in range(max_steps):
        reply = call_llm(SYSTEM, messages)
        # Detect a single-line JSON tool call
        m = re.search(r'\{\s*"tool"\s*:\s*"(CSV|SQL|STATS|PLOT)"\s*,\s*"args"\s*:\s*(\{.*?\})\s*\}\s*$', reply.strip(), re.S)
        if m:
            tool, args = m.group(1), json.loads(m.group(2))
            try:
                obs = dispatch_tool(tool, args, tools)
            except Exception as e:
                obs = f"TOOL-ERROR: {e}"
            messages.append({"role":"assistant","content": reply})
            messages.append({"role":"system","content": f"TOOL_RESULT:\n{obs}"})
            continue
        messages.append({"role":"assistant","content": reply})
        if any(k in reply.lower() for k in ["final answer", "here's the summary", "summary:"]):
            return reply
    return messages[-1]["content"]

import os, json
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

from tools.csv_tool import CSVTool
from tools.sql_tool import SQLTool
from tools.plot_tool import PlotTool
from tools.stats_tool import StatsTool

def call_llm(system: str, messages: list[dict], model: str = None, temperature: float = 0.2) -> str:
    provider = os.getenv("LLM_PROVIDER", "hf")
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
    elif provider == "hf":
        # Hugging Face Text Generation Inference or Inference API
        # Preferred: text-generation-inference endpoint via base URL (TGI-compatible)
        base = os.getenv("HF_BASE_URL")
        token = os.getenv("HF_TOKEN")
        mdl = model or os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
        # If HF_BASE_URL is provided, use a TGI-compatible /v1/chat/completions route
        if base:
            import requests
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            # Build OpenAI-style chat body for TGI compatibility
            body = {
                "model": mdl,
                "messages": [{"role":"system","content":system}, *messages],
                "temperature": float(os.getenv("HF_TEMPERATURE", temperature)),
            }
            url = base.rstrip('/') + "/v1/chat/completions"
            r = requests.post(url, json=body, headers=headers, timeout=60)
            r.raise_for_status()
            data = r.json()
            # Try OpenAI-like shape first
            if isinstance(data, dict) and data.get("choices"):
                return data["choices"][0]["message"]["content"]
            # Fallback to generic text field if provider differs
            return data.get("generated_text") or data.get("message") or ""
        else:
            # Fallback: use huggingface_hub InferenceClient for chat-like generation
            from huggingface_hub import InferenceClient
            if not token:
                raise RuntimeError("Set HF_TOKEN for Hugging Face provider or configure HF_BASE_URL.")
            client = InferenceClient(model=mdl, token=token)
            # Simple prompt flatten: concatenate messages with role tags
            def flatten(ms):
                parts = [f"System: {system}"]
                for m in ms:
                    parts.append(f"{m['role'].capitalize()}: {m['content']}")
                parts.append("Assistant:")
                return "\n\n".join(parts)
            prompt = flatten(messages)
            out = client.text_generation(prompt, temperature=float(os.getenv("HF_TEMPERATURE", temperature)), max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "512")))
            return out
    else:
        raise RuntimeError("Unsupported LLM_PROVIDER. Set LLM_PROVIDER=hf or LLM_PROVIDER=local.")

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

def _extract_tool_call(reply: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Robustly extract a JSON object like {"tool":"CSV","args":{...}} from free-form text.

    Handles nested braces in args by brace balancing and ignores braces inside strings.
    Returns (tool_name, args_dict) if found, else None.
    """
    start_key = '"tool"'
    start_idx = reply.find(start_key)
    if start_idx == -1:
        return None
    # Find the opening brace of the JSON object containing this key
    brace_start = reply.rfind('{', 0, start_idx)
    if brace_start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    end_idx = None
    for i in range(brace_start, len(reply)):
        ch = reply[i]
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
    if end_idx is None:
        return None
    import json
    try:
        obj = json.loads(reply[brace_start:end_idx+1])
        name = obj.get("tool")
        args = obj.get("args", {})
        if isinstance(name, str) and isinstance(args, dict):
            return name, args
    except Exception:
        return None
    return None

def run_agent(user_query: str, tools: ToolSuite, df_default_path: str = "data/sample_sales.csv", max_steps: int = 10) -> str:
    messages: list[dict] = [
        {"role":"user","content": f"""{user_query}
If you need data, use CSV.load('{df_default_path}')."""}
    ]
    for _ in range(max_steps):
        reply = call_llm(SYSTEM, messages)
        extracted = _extract_tool_call(reply)
        if extracted:
            t, args = extracted
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

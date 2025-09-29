# main.py — Streamlit/HF-friendly agent runtime with robust debugging & fallbacks
# - Better error messages (no more bare "ERROR:")
# - Hugging Face provider controls: HF_PROVIDER=hf-inference (recommended) or custom HF_BASE_URL (TGI/OpenAI-style)
# - Automatic fallback from text_generation → chat_completion when provider/model only supports "conversational"
# - Retries with backoff on transient errors/timeouts
# - DEBUG transcript logging, and optional RETURN_TRANSCRIPT to append a short trace into the final answer
# - Safer JSON tool-call extraction with brace balancing

import os
import json
import re
import time
import traceback
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

from tools.csv_tool import CSVTool
from tools.sql_tool import SQLTool
from tools.plot_tool import PlotTool
from tools.stats_tool import StatsTool

# ----------------------------
# Debug/Config toggles
# ----------------------------
DEBUG = os.getenv("DEBUG", "0") == "1"
RETURN_TRANSCRIPT = os.getenv("RETURN_TRANSCRIPT", "0") == "1"
MAX_STEPS = int(os.getenv("MAX_STEPS", "12"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))
RETRIES = int(os.getenv("RETRIES", "2"))
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "1.5"))

# ----------------------------
# Utility: structured debug prints
# ----------------------------
def dprint(*args):
    if DEBUG:
        print("[DEBUG]", *args, flush=True)

def _short(s: str, n: int = 400):
    s = s.strip()
    return s if len(s) <= n else s[:n] + "... [truncated]"

# ----------------------------
# Prompt flattening (for text-generation style models)
# ----------------------------
def _messages_to_prompt(system: str, messages: List[dict]) -> str:
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
            buf.append("### Tool/Context\n" + content)
        else:
            buf.append(f"### {role.capitalize()}\n" + content)
    buf.append("### Assistant")
    return "\n\n".join(buf).strip()

# ----------------------------
# LLM call with retries + HF fallbacks
# ----------------------------
def _call_hf_inference(messages: List[dict], system: str, model: str, token: str, provider: str) -> str:
    """
    Try text_generation first (works for most instruct models with hf-inference).
    If provider/model complains about 'conversational only', fall back to chat_completion.
    """
    from huggingface_hub import InferenceClient

    client = InferenceClient(model=model, token=token, provider=provider if provider else None)
    prompt = _messages_to_prompt(system, messages)
    max_new_tokens = int(float(os.getenv("HF_MAX_NEW_TOKENS", "512")))
    temperature = float(os.getenv("HF_TEMPERATURE", "0.2"))
    do_sample = temperature > 0

    # Try text_generation
    try:
        dprint("HF text_generation →", {"model": model, "provider": provider, "max_new_tokens": max_new_tokens, "temp": temperature})
        text = client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            repetition_penalty=1.05,
            return_full_text=False,
            timeout=REQUEST_TIMEOUT,
        )
        if isinstance(text, str) and text.strip():
            return text
        raise RuntimeError(f"Empty response from text_generation (model={model})")
    except Exception as e:
        emsg = f"{type(e).__name__}: {str(e)}"
        dprint("text_generation failed:", emsg)
        # If provider/model only supports conversational/chat, try chat_completion
        if "conversational" in emsg.lower() or "Supported task: conversational" in emsg or "chat" in emsg.lower():
            try:
                dprint("HF chat_completion fallback →", {"model": model, "provider": provider})
                # Convert messages to the HF chat format
                # HF expects: [{"role": "user"/"assistant"/"system", "content": "..."}, ...]
                chat_resp = client.chat_completion(
                    messages=[{"role": "system", "content": system}, *messages],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    timeout=REQUEST_TIMEOUT,
                )
                # huggingface_hub returns a dict-like object with choices -> message -> content (OpenAI-like)
                if hasattr(chat_resp, "choices") and chat_resp.choices:
                    c0 = chat_resp.choices[0]
                    # Support both dict and object attrs
                    msg = c0.get("message") if isinstance(c0, dict) else getattr(c0, "message", None)
                    if isinstance(msg, dict):
                        content = msg.get("content")
                    else:
                        content = getattr(msg, "content", None)
                    if content:
                        return content
                # Some providers may return plain text under .generated_text
                content = getattr(chat_resp, "generated_text", None) or (chat_resp.get("generated_text") if isinstance(chat_resp, dict) else None)
                if content:
                    return content
                raise RuntimeError("Empty response from chat_completion fallback")
            except Exception as e2:
                raise RuntimeError(f"chat_completion fallback failed: {type(e2).__name__}: {str(e2)}") from e2
        raise

def _call_hf_tgi(messages: List[dict], system: str, model: str, token: Optional[str], base_url: str) -> str:
    """
    Call a TGI/OpenAI-compatible /v1/chat/completions endpoint.
    """
    import requests
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, *messages],
        "temperature": float(os.getenv("HF_TEMPERATURE", "0.2")),
        "max_tokens": int(float(os.getenv("HF_MAX_NEW_TOKENS", "512"))),
    }
    url = base_url.rstrip("/") + "/v1/chat/completions"
    dprint("POST", url, "body_preview=", _short(json.dumps(body)[:600]))
    r = requests.post(url, json=body, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    # Try OpenAI-like shape first
    if isinstance(data, dict) and data.get("choices"):
        content = data["choices"][0]["message"]["content"]
        if content:
            return content
    # Fallback shapes
    for key in ("generated_text", "message", "text"):
        if key in data and data[key]:
            return data[key]
    raise RuntimeError(f"TGI response missing expected fields: keys={list(data)[:8]}")

def call_llm(system: str, messages: List[dict], model: Optional[str] = None, temperature: float = 0.2) -> str:
    """
    Unified LLM caller with:
      - provider switch: local | hf
      - hf modes: HF_BASE_URL (TGI/OpenAI-style) OR HF_PROVIDER=hf-inference
      - retries, better errors, and debug prints
    """
    provider = os.getenv("LLM_PROVIDER", "hf")
    last_err = None

    for attempt in range(1, RETRIES + 2):
        try:
            if provider == "local":
                import requests
                base = os.getenv("LLM_BASE_URL", "http://localhost:8000")
                mdl = os.getenv("LLM_MODEL", model or "llama3.1:8b-instruct")
                body = {
                    "model": mdl,
                    "messages": [{"role": "system", "content": system}, *messages],
                    "temperature": float(os.getenv("LOCAL_TEMPERATURE", temperature)),
                }
                url = base.rstrip("/") + "/chat"
                dprint(f"[{attempt}/{RETRIES+1}] local LLM POST", url, "body_preview=", _short(json.dumps(body)[:600]))
                resp = requests.post(url, json=body, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                msg = data.get("message") or data.get("generated_text") or ""
                if not msg:
                    raise RuntimeError("Local provider returned empty message")
                return msg

            elif provider == "hf":
                mdl = model or os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
                token = os.getenv("HF_TOKEN")
                base = os.getenv("HF_BASE_URL")  # if set → use TGI/OpenAI-style
                prov = os.getenv("HF_PROVIDER", "hf-inference")  # recommended default

                if base:
                    dprint(f"[{attempt}/{RETRIES+1}] HF TGI route → base={base}, model={mdl}")
                    return _call_hf_tgi(messages, system, mdl, token, base_url=base)
                else:
                    dprint(f"[{attempt}/{RETRIES+1}] HF Inference route → provider={prov}, model={mdl}")
                    if not token:
                        raise RuntimeError("HF_TOKEN not set for hf inference provider")
                    return _call_hf_inference(messages, system, mdl, token, provider=prov)

            else:
                raise RuntimeError("Unsupported LLM_PROVIDER. Use LLM_PROVIDER=hf or local.")

        except Exception as e:
            last_err = e
            dprint(f"Attempt {attempt} failed:", f"{type(e).__name__}: {str(e)}")
            if attempt <= RETRIES:
                sleep_s = RETRY_BACKOFF ** attempt
                dprint(f"Retrying in ~{sleep_s:.1f}s ...")
                time.sleep(sleep_s)
            else:
                # Surface rich error
                raise RuntimeError(
                    f"LLM call failed after {RETRIES+1} attempts. "
                    f"Last error: {type(e).__name__}: {str(e)}\n{traceback.format_exc(limit=2)}"
                ) from e

# ----------------------------
# System prompt
# ----------------------------
SYSTEM = """You are a careful data-analysis agent that plans with a ReAct loop.
When you need a tool, emit EXACTLY one JSON line:
{"tool":"CSV","args":{"cmd":"load","path":"data/sample_sales.csv"}}
Valid tools: CSV, SQL, STATS, PLOT.
After you get TOOL_RESULT, continue planning or finish with a short, clear summary.
List any saved plot paths in the final answer.
"""

# ----------------------------
# Tools plumbing
# ----------------------------
@dataclass
class ToolSuite:
    csv: CSVTool
    sql: SQLTool
    plot: PlotTool
    stats: StatsTool

def dispatch_tool(name: str, args: Dict[str, Any], tools: ToolSuite) -> str:
    if name == "CSV":
        cmd = args.get("cmd", "load")
        if cmd == "load":
            return tools.csv.load(args["path"])
        if cmd == "head":
            return tools.csv.head(args.get("n", 5))
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
        cmd = args.get("cmd", "query")
        if cmd == "load_from_df":
            return tools.sql.load_from_df(tools.csv.get_df(), args.get("table_name", "df"))
        if cmd == "query":
            return tools.sql.query(args["sql"], args.get("limit", 50))
        raise ValueError(f"Unknown SQL cmd: {cmd}")
    if name == "STATS":
        cmd = args.get("cmd", "corr")
        df = tools.csv.get_df()
        if cmd == "corr":
            return tools.stats.corr(df, args["cols"])
        if cmd == "ttest":
            return tools.stats.ttest(df, args["col"], args["by"])
        if cmd == "z_anomalies":
            return tools.stats.z_anomalies(df[args["col"]], args.get("threshold", 3.0))
        raise ValueError(f"Unknown STATS cmd: {cmd}")
    if name == "PLOT":
        cmd = args.get("cmd", "line")
        df = tools.csv.get_df()
        if cmd == "line":
            return tools.plot.line(df, args["x"], args["y"], args.get("title", ""))
        if cmd == "bar":
            return tools.plot.bar(df, args["x"], args["y"], args.get("title", ""))
        if cmd == "hist":
            return tools.plot.hist(df[args["col"]], args.get("bins", 20), args.get("title", ""))
        raise ValueError(f"Unknown PLOT cmd: {cmd}")
    raise ValueError(f"Unknown tool: {name}")

# ----------------------------
# Robust tool-call extraction
# ----------------------------
def _extract_tool_call(reply: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Extract {"tool":"CSV","args":{...}} from free-form text using brace balancing.
    Ignores braces inside strings. Returns (tool_name, args_dict) or None.
    """
    if not reply:
        return None
    start_key = '"tool"'
    start_idx = reply.find(start_key)
    if start_idx == -1:
        return None
    brace_start = reply.rfind("{", 0, start_idx)
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
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
    if end_idx is None:
        return None
    try:
        obj = json.loads(reply[brace_start : end_idx + 1])
        name = obj.get("tool")
        args = obj.get("args", {})
        if isinstance(name, str) and isinstance(args, dict):
            return name, args
    except Exception as e:
        dprint("JSON parse failed in _extract_tool_call:", f"{type(e).__name__}: {str(e)}")
    return None

# ----------------------------
# Agent loop
# ----------------------------
def run_agent(user_query: str, tools: ToolSuite, df_default_path: str = "data/sample_sales.csv", max_steps: int = MAX_STEPS) -> str:
    messages: List[dict] = [
        {
            "role": "user",
            "content": f"""{user_query}
If you need data, use CSV.load('{df_default_path}').""",
        }
    ]
    transcript_chunks: List[str] = []

    for step in range(1, max_steps + 1):
        dprint(f"STEP {step} → calling LLM; last_user={_short(messages[-1]['content'] if messages else '')}")
        try:
            reply = call_llm(SYSTEM, messages)
        except Exception as e:
            err = f"LLM_ERROR step={step}: {type(e).__name__}: {str(e)}"
            dprint(err)
            raise

        dprint(f"STEP {step} ← LLM reply preview:", _short(reply))
        transcript_chunks.append(f"[Assistant {step}] { _short(reply, 800) }")

        extracted = _extract_tool_call(reply)
        if extracted:
            t, args = extracted
            dprint(f"STEP {step} → TOOL CALL:", t, args)
            try:
                obs = dispatch_tool(t, args, tools)
                dprint(f"STEP {step} ← TOOL RESULT preview:", _short(obs))
            except Exception as e:
                obs = f"TOOL-ERROR: {type(e).__name__}: {str(e)}"
                dprint(f"STEP {step} TOOL ERROR:", obs)
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "system", "content": f"TOOL_RESULT:\n{obs}"})
            transcript_chunks.append(f"[Tool {t}] { _short(obs, 800) }")
            continue

        messages.append({"role": "assistant", "content": reply})

        # Heuristics to detect finish
        if any(k in reply.lower() for k in ["final answer", "here's the summary", "summary:"]):
            final = reply
            if RETURN_TRANSCRIPT:
                final += "\n\n---\n[debug transcript]\n" + "\n".join(transcript_chunks[-8:])
            return final

    # Fallback: return last assistant reply if we hit step limit
    final = messages[-1]["content"] if messages else ""
    if RETURN_TRANSCRIPT:
        final += "\n\n---\n[debug transcript]\n" + "\n".join(transcript_chunks[-8:])
    return final

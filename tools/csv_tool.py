from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from typing import Optional, Dict, Any

@dataclass
class CSVState:
    df: Optional[pd.DataFrame] = None
    path: Optional[str] = None

class CSVTool:
    name = "CSV"
    desc = "Work with CSV/XLSX via pandas: load, head, info, describe, filter, groupby."

    def __init__(self):
        self.state = CSVState()

    def load(self, path: str) -> str:
        if path.lower().endswith((".xlsx", ".xls")):
            self.state.df = pd.read_excel(path)
        else:
            self.state.df = pd.read_csv(path)
        self.state.path = path
        return f"Loaded file with shape {self.state.df.shape}. Columns: {list(self.state.df.columns)}"

    def head(self, n: int = 5) -> str:
        self._require_df()
        return self.state.df.head(n).to_markdown(index=False)

    def info(self) -> str:
        self._require_df()
        buf = []
        buf.append(f"Shape: {self.state.df.shape}")
        buf.append("Dtypes:")
        buf.append(self.state.df.dtypes.to_string())
        buf.append("Nulls (sum):")
        buf.append(self.state.df.isna().sum().to_string())
        return "\n".join(buf)

    def describe(self) -> str:
        self._require_df()
        return self.state.df.describe(include='all').to_markdown()

    def groupby_agg(self, by: list, agg_map: Dict[str, Any]) -> str:
        self._require_df()
        g = self.state.df.groupby(by).agg(agg_map).reset_index()
        self.state.df = g
        return g.head(20).to_markdown(index=False)

    def filter_query(self, expr: str) -> str:
        self._require_df()
        filtered = self.state.df.query(expr)
        return filtered.head(20).to_markdown(index=False)

    def get_df(self) -> pd.DataFrame:
        self._require_df()
        return self.state.df

    def set_df(self, df: pd.DataFrame):
        self.state.df = df

    def _require_df(self):
        if self.state.df is None:
            raise RuntimeError("No DataFrame loaded. Use CSV.load(path) first.")

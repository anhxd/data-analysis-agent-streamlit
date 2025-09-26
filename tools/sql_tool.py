from __future__ import annotations
import pandas as pd
import sqlite3
from dataclasses import dataclass

@dataclass
class SQLState:
    conn: sqlite3.Connection | None = None
    table_name: str = "df"

class SQLTool:
    name = "SQL"
    desc = "Run SQL over current DataFrame via temporary SQLite mirror."

    def __init__(self):
        self.state = SQLState()

    def load_from_df(self, df: pd.DataFrame, table_name: str = "df") -> str:
        self.state.conn = sqlite3.connect(":memory:")
        df.to_sql(table_name, self.state.conn, if_exists="replace", index=False)
        self.state.table_name = table_name
        return f"Loaded DataFrame into SQLite as table '{table_name}' with {len(df)} rows."

    def query(self, sql: str, limit: int = 50) -> str:
        if self.state.conn is None:
            raise RuntimeError("No SQLite connection. Call load_from_df(df) first.")
        df = pd.read_sql_query(sql, self.state.conn)
        return df.head(limit).to_markdown(index=False)

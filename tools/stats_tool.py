import pandas as pd
import numpy as np
from scipy import stats

class StatsTool:
    name = "STATS"
    desc = "Quick stats: correlations, two-sample t-tests, z-score anomalies."

    def corr(self, df: pd.DataFrame, cols: list[str]) -> str:
        corr = df[cols].corr(numeric_only=True)
        return corr.to_markdown()

    def ttest(self, df: pd.DataFrame, col: str, by: str) -> str:
        groups = [g[col].dropna().values for _, g in df.groupby(by)]
        if len(groups) != 2:
            return "T-test requires exactly 2 groups."
        t, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        return f"t={t:.3f}, p={p:.6f}"

    def z_anomalies(self, series: pd.Series, threshold: float = 3.0) -> str:
        z = np.abs((series - series.mean()) / (series.std(ddof=0) + 1e-9))
        idx = np.where(z > threshold)[0][:20]
        return f"Found {len(idx)} anomalies at indices (first 20): {idx.tolist()}"

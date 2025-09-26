import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

class PlotTool:
    name = "PLOT"
    desc = "Create simple plots (line, bar, hist) and save to outputs/."

    def __init__(self, out_dir: str = "outputs"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def line(self, df: pd.DataFrame, x: str, y: str, title: str = "") -> str:
        fig = plt.figure()
        df.plot(x=x, y=y)
        if title:
            plt.title(title)
        path = self._save(fig, "line")
        return f"Saved line plot to {path}"

    def bar(self, df: pd.DataFrame, x: str, y: str, title: str = "") -> str:
        fig = plt.figure()
        df.plot(kind="bar", x=x, y=y)
        if title:
            plt.title(title)
        path = self._save(fig, "bar")
        return f"Saved bar plot to {path}"

    def hist(self, series: pd.Series, bins: int = 20, title: str = "") -> str:
        fig = plt.figure()
        series.plot(kind="hist", bins=bins)
        if title:
            plt.title(title)
        path = self._save(fig, "hist")
        return f"Saved histogram to {path}"

    def _save(self, fig, prefix: str) -> str:
        import matplotlib
        matplotlib.use("Agg")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(self.out_dir, f"{prefix}_{ts}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

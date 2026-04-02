from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path


class BaseExperiment(ABC):
    def __init__(self, config: dict, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.results_dir = output_dir / "results"
        self.plots_dir = output_dir / "plots"
        self.logs_dir = output_dir / "logs"
        self.plot_format = str(config.get("output", {}).get("plot_format", "png")).lower()
        if self.plot_format not in {"png", "pdf"}:
            raise ValueError(f"Unsupported output.plot_format: {self.plot_format!r}")
        for path in (self.output_dir, self.results_dir, self.plots_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError

    def save_json(self, payload: dict, filename: str) -> Path:
        path = self.results_dir / filename
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def save_csv(self, rows: list[dict], filename: str) -> Path:
        path = self.results_dir / filename
        if not rows:
            path.write_text("", encoding="utf-8")
            return path
        fieldnames: list[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key in seen:
                    continue
                seen.add(key)
                fieldnames.append(key)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with (self.logs_dir / "experiment.log").open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")

    def plot_filename(self, stem: str) -> str:
        return f"{stem}.{self.plot_format}"

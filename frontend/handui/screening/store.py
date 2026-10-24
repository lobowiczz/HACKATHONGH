"""Append-only JSON-lines log of finished tests, kept on this machine only."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import DATA_DIR

RESULTS_PATH = DATA_DIR / "screening" / "results.jsonl"


class ResultStore:
    def __init__(self, path: Path = RESULTS_PATH):
        self.path = path

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def load(self, source: str | None = None) -> list[dict]:
        """All records, oldest first. A damaged line is skipped, not fatal."""
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if source is None or rec.get("source") == source:
                records.append(rec)
        return records

    def history(self, source: str | None = None) -> list[dict]:
        """Records without their raw series, light enough to list in the dashboard."""
        return [{k: v for k, v in r.items() if k != "series"} for r in self.load(source)]

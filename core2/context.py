# core2/context.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any
import pandas as pd

@dataclass
class Context:
    cfg: Dict[str, Any]
    df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    meta: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def require_cols(self, cols):
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
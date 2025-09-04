# core2/context.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

import pandas as pd
# Import kept for typing reference (no inheritance needed at runtime)
from rs3_contracts.api import ContextSpec  # noqa: F401


@dataclass
class Context:
    """Concrete runtime context carried across RS3 stages.

    Matches the `ContextSpec` protocol structurally (cfg, meta, set_meta).
    """

    cfg: Dict[str, Any]
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    meta: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    # ==== Contract method (ContextSpec) ====
    def set_meta(self, key: str, value: Any) -> None:
        self.meta[key] = value

    # ==== Helpers ====
    def get_meta(self, key: str, default: Any | None = None) -> Any:
        return self.meta.get(key, default)

    def update_meta(self, **kwargs: Any) -> None:
        self.meta.update(kwargs)

    def add_artifact(self, name: str, value: Any) -> None:
        self.artifacts[name] = value

    def require_cols(self, cols: Iterable[str]) -> None:
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
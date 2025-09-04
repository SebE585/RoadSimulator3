# core2/pipeline.py
from __future__ import annotations
from typing import List
from .contracts import Result
from .context import Context

class PipelineSimulator:
    def __init__(self, stages: List[object]):
        self.stages = stages

    def run(self, ctx: Context) -> Context:
        for st in self.stages:
            name = getattr(st, "name", st.__class__.__name__)
            res: Result = st.run(ctx)
            if not isinstance(res, Result):
                raise RuntimeError(f"[{name}] returned invalid result type")
            if not res.ok:
                raise RuntimeError(f"[{name}] {res.message}")
        return ctx
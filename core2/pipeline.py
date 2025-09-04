# core2/pipeline.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List

from rs3_contracts.api import ContextSpec, Result, Stage

logger = logging.getLogger(__name__)


@dataclass
class PipelineSimulator:
    """Minimal pipeline runner for RS3 based on the public contracts.

    Parameters
    ----------
    stages : list[Stage]
        Ordered list of stages to execute.
    stop_on_error : bool
        If True, stops the pipeline at first failing stage; otherwise continues and
        accumulates errors but returns a failing Result at the end.
    """

    stages: List[Stage]
    stop_on_error: bool = True

    def run(self, ctx: ContextSpec) -> Result:
        """Execute all stages sequentially.

        Notes
        -----
        - Each stage must implement `run(ContextSpec) -> Result`.
        - Any unexpected exception is caught and converted to a failing `Result`.
        """
        pipeline_name = getattr(ctx, "meta", {}).get("pipeline_name", "rs3-pipeline")
        start_ts = time.time()
        logger.info("[PIPELINE] %s — %d stages", pipeline_name, len(self.stages))

        any_error = False
        for i, stage in enumerate(self.stages, start=1):
            t0 = time.time()
            stage_name = getattr(stage, "name", stage.__class__.__name__)
            logger.info("[STAGE %d/%d] %s — start", i, len(self.stages), stage_name)
            try:
                res = stage.run(ctx)
                res = Result(res)
                # Ensure failing stages always carry a message to ease debugging
                if not res.ok:
                    # Si le message est vide ou whitespace, on ajoute un message par défaut
                    if not isinstance(res.msg, str) or not res.msg.strip():
                        res = Result((False, f"{stage_name} failed without message"))
                    logger.debug(
                        "[STAGE %d/%d] %s — raw result normalized to: ok=%s msg=%r",
                        i,
                        len(self.stages),
                        stage_name,
                        res.ok,
                        res.msg,
                    )
            except Exception as exc:  # noqa: BLE001 — convert to Result
                logger.exception(
                    "[STAGE %d/%d] %s — CRASH: %s", i, len(self.stages), stage_name, exc
                )
                res = Result((False, f"Exception in stage {stage_name}: {exc}"))

            dt_ms = (time.time() - t0) * 1000
            if res.ok:
                logger.info(
                    "[STAGE %d/%d] %s — OK (%.1f ms)", i, len(self.stages), stage_name, dt_ms
                )
            else:
                any_error = True
                logger.error(
                    "[STAGE %d/%d] %s — FAIL: %s (%.1f ms)",
                    i,
                    len(self.stages),
                    stage_name,
                    res.msg,
                    dt_ms,
                )
                if self.stop_on_error:
                    total_ms = (time.time() - start_ts) * 1000
                    logger.error(
                        "[PIPELINE] %s — ABORT after failure (%.1f ms)", pipeline_name, total_ms
                    )
                    return Result((False, f"Abort after stage failure: {res.msg}"))

        total_ms = (time.time() - start_ts) * 1000
        if any_error:
            logger.error("[PIPELINE] %s — DONE with errors (%.1f ms)", pipeline_name, total_ms)
            return Result((False, "One or more stages failed"))

        logger.info("[PIPELINE] %s — DONE (%.1f ms)", pipeline_name, total_ms)
        return Result((True, "OK"))


def build_pipeline(*stages: Stage, stop_on_error: bool = True) -> PipelineSimulator:
    """Small helper to build a `PipelineSimulator` with a fluent API.

    Example
    -------
    >>> pipe = build_pipeline(FetchOSRM(), InjectNoise(), ExportCSV())
    >>> res = pipe.run(ctx)
    >>> assert res.ok
    """
    return PipelineSimulator(stages=list(stages), stop_on_error=stop_on_error)

if __name__ == "__main__":
    # Simple demo when running this file directly
    class _OkStage(Stage):
        name = "ok"
        def run(self, ctx: ContextSpec) -> Result:
            return Result((True, "OK"))

    demo_ctx = {"meta": {"pipeline_name": "demo"}}
    pipe = build_pipeline(_OkStage())
    res = pipe.run(demo_ctx)  # type: ignore[arg-type]
    print("Demo result:", res.ok, res.msg)
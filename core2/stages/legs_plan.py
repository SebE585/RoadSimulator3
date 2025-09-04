# core2/stages/legs_plan.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from rs3_contracts.api import ContextSpec, Result, Stage
from ..context import Context

@dataclass
class Stop:
    id: str
    lat: float
    lon: float
    service_s: int = 0
    tw_start: Optional[datetime] = None
    tw_end: Optional[datetime] = None

class LegsPlan:
    """
    Construit la séquence de legs (A->B) à partir d'une liste de stops ordonnés.
    """
    name = "LegsPlan"

    @staticmethod
    def _parse_dt(x: Optional[str]) -> Optional[datetime]:
        if not x:
            return None
        # accepte "...Z" ou "...+00:00"
        return datetime.fromisoformat(x.replace("Z", "+00:00"))

    def run(self, ctx: Context) -> Result:
        vcfg = ctx.cfg
        stops_cfg: List[Dict[str, Any]] = vcfg.get("stops", [])
        if len(stops_cfg) < 2:
            return Result((False, "Au moins 2 stops requis (départ et arrivée)."))

        stops: List[Stop] = []
        for s in stops_cfg:
            stops.append(Stop(
                id=str(s.get("id", "")),
                lat=float(s["lat"]), lon=float(s["lon"]),
                service_s=int(s.get("service_s", 0)),
                tw_start=self._parse_dt(s.get("tw_start")),
                tw_end=self._parse_dt(s.get("tw_end")),
            ))

        legs = []
        for i in range(len(stops) - 1):
            legs.append({
                "from": stops[i].__dict__,
                "to":   stops[i+1].__dict__,
                "idx": i
            })

        start_time = vcfg.get("start_time_utc")
        if start_time:
            t0 = self._parse_dt(start_time)  # timezone-aware if input had Z/+00:00
        else:
            # défaut: maintenant en UTC, timezone-aware
            t0 = datetime.now(timezone.utc)

        plan = {
            "stops": [s.__dict__ for s in stops],
            "legs": legs,
            "start_time_utc": t0.isoformat().replace("+00:00", "Z")
        }
        ctx.artifacts["legs_plan"] = plan
        return Result((True, "OK"))
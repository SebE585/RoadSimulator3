# core2/stages/stopwait_injector.py
from __future__ import annotations
import pandas as pd
from datetime import timedelta
from rs3_contracts.api import ContextSpec, Result, Stage
from ..context import Context

class StopWaitInjector:
    """
    Ajoute:
      - WAIT (arrivée avant fenêtre) : clamp v=0 jusqu'à tw_start
      - STOP (service_s) : clamp v=0 pendant service
    """
    name = "StopWaitInjector"

    def run(self, ctx: Context) -> Result:
        plan = ctx.artifacts.get("legs_plan")
        summaries = ctx.artifacts.get("legs_summary", [])
        df = ctx.df
        hz = int(ctx.meta.get("hz", 10))

        if not plan or df is None or df.empty:
            return Result(ok=False, message="legs_plan/df manquant")

        stops = plan["stops"]
        t0 = pd.to_datetime(df["timestamp"].iloc[0], utc=True)

        def t_to_idx(t: pd.Timestamp) -> int:
            sec = (t - t0).total_seconds()
            return max(0, min(len(df) - 1, int(round(sec * hz))))

        # Bâtit un calendrier relatif aux durées des legs
        schedule_rows = []
        wait_intervals = []
        stop_intervals = []

        cursor_t = pd.to_datetime(plan.get("start_time_utc", str(t0)), utc=True)

        for i, stop in enumerate(stops):
            if i == 0:
                schedule_rows.append({
                    "stop_id": stop["id"],
                    "arrive": cursor_t,
                    "wait_s": 0,
                    "service_s": 0,
                    "depart": cursor_t
                })
                continue

            leg_sum = next((x for x in summaries if x["idx"] == i-1), None)
            travel_s = float(leg_sum["duration_s"]) if leg_sum else 0.0
            arrive_t = cursor_t + timedelta(seconds=travel_s)

            tw_start = pd.to_datetime(stop["tw_start"], utc=True) if stop.get("tw_start") else None
            service_s = int(stop.get("service_s", 0))

            if tw_start and arrive_t < tw_start:
                wait_s = (tw_start - arrive_t).total_seconds()
                wait_intervals.append((arrive_t, tw_start))
                start_service_t = tw_start
            else:
                wait_s = 0
                start_service_t = arrive_t

            if service_s > 0:
                stop_start = start_service_t
                stop_end = stop_start + timedelta(seconds=service_s)
                stop_intervals.append((stop_start, stop_end))
                depart_t = stop_end
            else:
                depart_t = start_service_t

            schedule_rows.append({
                "stop_id": stop["id"],
                "arrive": arrive_t,
                "wait_s": int(wait_s),
                "service_s": service_s,
                "depart": depart_t
            })
            cursor_t = depart_t

        df = df.copy()
        df["flag_wait"] = 0
        df["flag_stop"] = 0

        for (a, b) in wait_intervals:
            i0, i1 = t_to_idx(a), t_to_idx(b)
            df.loc[i0:i1, "flag_wait"] = 1

        for (a, b) in stop_intervals:
            i0, i1 = t_to_idx(a), t_to_idx(b)
            df.loc[i0:i1, "flag_stop"] = 1

        # clamp speed
        clamp = (df["flag_wait"] == 1) | (df["flag_stop"] == 1)
        df.loc[clamp, "speed"] = 0.0

        ctx.df = df
        # sérialisation lisible
        ctx.artifacts["stops_schedule"] = [
            {
                "stop_id": r["stop_id"],
                "arrive": r["arrive"].isoformat(),
                "wait_s": r["wait_s"],
                "service_s": r["service_s"],
                "depart": r["depart"].isoformat()
            } for r in schedule_rows
        ]
        ctx.artifacts["wait_intervals"] = [(a.isoformat(), b.isoformat()) for (a, b) in wait_intervals]
        ctx.artifacts["stop_intervals"] = [(a.isoformat(), b.isoformat()) for (a, b) in stop_intervals]
        return Result()
# core2/stages/validators.py
from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd
import numpy as np

from ..contracts import Result
from ..context import Context

logger = logging.getLogger(__name__)



def _compute_checklist(df: pd.DataFrame, hz_target: float = 10.0) -> tuple[dict, dict]:
    """Calcule une checklist standardisée (✅/❌) et quelques métriques.
    Retourne (checks, metrics).
    """
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}

    # Cadence & timeline
    t = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    ns = t.astype("int64").to_numpy()
    dt = np.diff((ns - ns[0]) / 1e9, prepend=0.0)
    pos = dt > 0
    if not pos.any():
        checks["cadence_10hz"] = False
        metrics["dt_median_s"] = float("nan")
        metrics["hz_obs"] = float("nan")
        # Même si cadence KO, on continue à remplir le reste prudemment
        dt_med = float("nan")
    else:
        dt_med = float(np.median(dt[pos]))
        metrics["dt_median_s"] = dt_med
        metrics["hz_obs"] = (1.0 / dt_med) if dt_med > 0 else float("nan")
        checks["cadence_10hz"] = abs(dt_med - 1.0 / hz_target) < 0.02  # ±20 ms

    # Vitesse (m/s et km/h)
    sp = pd.to_numeric(df.get("speed", 0.0), errors="coerce").fillna(0.0).to_numpy()
    if "speed_kmh" in df.columns:
        sp_kmh = pd.to_numeric(df["speed_kmh"], errors="coerce").fillna(0.0).to_numpy()
        sp_mps = sp_kmh / 3.6
    else:
        sp_mps = sp
    checks["speed_nonnegative"] = bool((sp_mps >= -1e-6).all())

    # Départ/fin à 0 (médiane sur 1 s)
    n_tail = max(1, int(round(hz_target * 1.0)))  # 1 s window
    start0 = float(np.nanmedian(sp_mps[:n_tail])) < 0.6  # 0.6 m/s ≈ 2.16 km/h
    end0 = float(np.nanmedian(sp_mps[-n_tail:])) < 0.6
    checks["start_zero"] = start0
    checks["end_zero"] = end0

    # Variabilité inertielle minimale (sur mouvement)
    moving = sp_mps > 0.5
    v_med = float(np.nanmedian(sp_mps[moving])) if moving.any() else 0.0
    metrics["v_median_mps"] = v_med

    ax = pd.to_numeric(df.get("acc_x", 0.0), errors="coerce").fillna(0.0).to_numpy()
    gz = pd.to_numeric(df.get("gyro_z", 0.0), errors="coerce").fillna(0.0).to_numpy()
    std_ax = float(np.std(ax[moving])) if moving.any() else 0.0
    std_gz = float(np.std(gz[moving])) if moving.any() else 0.0
    metrics["std_ax"] = std_ax
    metrics["std_gz"] = std_gz
    checks["ax_variability"] = std_ax > 0.02   # m/s²
    checks["gz_variability"] = std_gz > 0.002  # rad/s

    # Cohérence latérale sur virages (ay ≈ v * gz) + rayon plausible
    ay = pd.to_numeric(df.get("acc_y", 0.0), errors="coerce").fillna(0.0).to_numpy()
    turn = np.logical_and(moving, np.abs(gz) > 0.01)
    if turn.any():
        # 1) Cohérence latérale: ay ≈ v * gz
        ay_pred = sp_mps[turn] * gz[turn]
        err = float(np.nanmedian(np.abs(ay[turn] - ay_pred)))
        metrics["lat_err_mps2_med"] = err
        checks["lateral_consistency"] = err < 0.5  # tolérance m/s²

        # 2) Rayon de courbure plausible
        #   a) basé sur le gyro (plus robuste): R_gz ≈ v / |gz|
        R_gz = sp_mps[turn] / np.maximum(np.abs(gz[turn]), 1e-3)
        plausible_gz = np.logical_and(R_gz > 10.0, R_gz < 3000.0)
        ratio_gz = float(np.mean(plausible_gz)) if R_gz.size else 1.0
        metrics["turn_radius_from_gz_ratio"] = ratio_gz
        metrics["turn_radius_from_gz_med"] = float(np.nanmedian(R_gz)) if R_gz.size else float("nan")

        #   b) à titre indicatif: rayon depuis ay (peut être bruité/lissé)
        R_ay = (sp_mps[turn] ** 2) / np.maximum(np.abs(ay[turn]), 1e-3)
        plausible_ay = np.logical_and(R_ay > 10.0, R_ay < 3000.0)
        ratio_ay = float(np.mean(plausible_ay)) if R_ay.size else 1.0
        metrics["turn_radius_from_ay_ratio"] = ratio_ay
        metrics["turn_radius_from_ay_med"] = float(np.nanmedian(R_ay)) if R_ay.size else float("nan")

        # On décide du check final sur le critère gyro (plus robuste)
        checks["turn_radius_plausible"] = ratio_gz > 0.7
    else:
        checks["lateral_consistency"] = True
        checks["turn_radius_plausible"] = True
        metrics["lat_err_mps2_med"] = 0.0
        metrics["turn_radius_from_gz_ratio"] = 1.0
        metrics["turn_radius_from_gz_med"] = float("nan")
        metrics["turn_radius_from_ay_ratio"] = 1.0
        metrics["turn_radius_from_ay_med"] = float("nan")

    return checks, metrics


def _realism_lite(df: pd.DataFrame, hz_target: float = 10.0) -> dict:
    """Vérifications de réalisme légères, sans dépendre de check_realism externe.
    Retourne un dict avec les clés: available, ok, checks, summary.
    """
    out: dict = {"available": True, "ok": True, "checks": {}}

    checks, metrics = _compute_checklist(df, hz_target=hz_target)
    out["checks"] = checks
    out["metrics"] = metrics

    crits = [
        "cadence_10hz",
        "start_zero",
        "end_zero",
        "speed_nonnegative",
        "ax_variability",
        "gz_variability",
        "lateral_consistency",
        "turn_radius_plausible",
    ]
    failed = [k for k in crits if not bool(checks.get(k, True))]
    out["failed"] = failed
    out["ok"] = len(failed) == 0
    out["summary"] = "OK" if not failed else "KO: " + ", ".join(failed)
    return out


class Validators:
    """
    Valide la cohérence du DataFrame final et exécute les vérifications de réalisme *intégrées* (sans dépendre de check/check_realism.py).
    Un rapport détaillé est stocké dans `ctx.artifacts['qa_realism']`.

    Configuration optionnelle via `cfg.validation` :
      validation:
        fail_on_realism: false   # stoppe la pipeline si réalisme NOK
        pass_basic_checks: true  # si false, on n'applique pas les checks de base
    """
    name = "Validators"

    def run(self, ctx: Context) -> Result:
        df = ctx.df
        if df is None or df.empty:
            return Result(ok=False, message="df vide")

        vcfg: Dict[str, Any] = {}
        if isinstance(ctx.cfg, dict):
            vcfg = ctx.cfg.get("validation", {}) or {}

        fail_on_realism: bool = bool(vcfg.get("fail_on_realism", False))
        pass_basic: bool = bool(vcfg.get("pass_basic_checks", True))

        # -------------------------
        # 1) Checks de base (légers)
        # -------------------------
        if pass_basic:
            # Vérifie présence colonne timestamp
            if "timestamp" not in df.columns:
                return Result(ok=False, message="timestamp manquant")
            # Parsing UTC (ne crée aucune nouvelle timeline)
            ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            # Invalid datetimes ?
            if ts.isna().any():
                n_bad = int(ts.isna().sum())
                logger.warning("Timestamps invalides détectés: %d", n_bad)
                if vcfg.get("fail_on_nan", False):
                    return Result(ok=False, message=f"{n_bad} timestamps invalides")
            # Monotonicité
            if not ts.is_monotonic_increasing:
                return Result(ok=False, message="timestamps non monotones")
            # NaN globaux (autres colonnes)
            if df.isnull().any().any():
                logger.warning("NaN détectés dans le DataFrame de sortie.")
                ctx.artifacts["qa_basic"] = {"nan_detected": True}
                if vcfg.get("fail_on_nan", False):
                    return Result(ok=False, message="NaN détectés")
            else:
                ctx.artifacts["qa_basic"] = {"nan_detected": False}

        # -------------------------
        # Guarantee IMU columns required by realism checker
        required_numeric_zeros = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        df = df.copy()
        for col in required_numeric_zeros:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        # -------------------------
        # Ensure required columns for check_realism
        # At minimum, 'event' is expected by the checker; create it if missing
        if "event" not in df.columns:
            df = df.copy()
            df["event"] = ""
        # propagate any additions back to context (Exporter will cast per schema)
        ctx.df = df
        # 2) Check réalisme (intégré uniquement)
        try:
            qa_realism: Dict[str, Any] = _realism_lite(df, hz_target=float(ctx.meta.get("hz", 10)))
        except Exception as e:
            logger.exception("realism_lite a échoué: %s", e)
            qa_realism = {"available": True, "ok": False, "error": str(e), "summary": "KO"}

        # --- Ajoute toujours la checklist standardisée pour affichage (✅/❌) ---
        try:
            std_checks, std_metrics = _compute_checklist(df, hz_target=float(ctx.meta.get("hz", 10)))
            # fusion non destructive
            chk = qa_realism.get("checks") or {}
            if not isinstance(chk, dict):
                chk = {}
            chk.update(std_checks)
            qa_realism["checks"] = chk
            mets = qa_realism.get("metrics") or {}
            if not isinstance(mets, dict):
                mets = {}
            for k, v in std_metrics.items():
                mets.setdefault(k, v)
            qa_realism["metrics"] = mets
        except Exception as e:
            logger.debug("checklist standard non ajoutée: %s", e)

        def _summarize_failure(d: dict) -> str:
            # prefer explicit 'failed' list
            failed = d.get("failed")
            if isinstance(failed, (list, tuple)) and failed:
                return ", ".join(map(str, failed[:5]))
            # else derive from boolean checks map if present
            checks = d.get("checks")
            if isinstance(checks, dict):
                bad = [k for k, v in checks.items() if v is False]
                if bad:
                    return ", ".join(bad[:5])
            # else fall back to errors/warnings/messages
            for key in ("errors", "warnings", "issues", "messages"):
                val = d.get(key)
                if isinstance(val, (list, tuple)) and val:
                    return ", ".join(map(str, val[:3]))
                if isinstance(val, str) and val:
                    return val
            return "raison inconnue"

        # Stocke un résumé lisible et explicite
        # Si 'ok' absent, dérive-le depuis la checklist standard
        if "ok" not in qa_realism:
            failed_from_checks = [k for k, v in (qa_realism.get("checks") or {}).items() if v is False]
            qa_realism["failed"] = failed_from_checks
            qa_realism["ok"] = len(failed_from_checks) == 0

        if qa_realism.get("ok", True):
            summary = "OK"
        else:
            reason = _summarize_failure(qa_realism)
            # si rien d'explicite, tente avec checklist fusionnée
            if reason == "raison inconnue":
                bad = [k for k, v in (qa_realism.get("checks") or {}).items() if v is False]
                if bad:
                    reason = ", ".join(bad[:5])
            summary = f"KO: {reason}"
            logger.warning("Realism KO — causes: %s", reason)
        qa_realism["summary"] = summary

        ctx.artifacts["qa_realism"] = qa_realism
        ctx.artifacts["qa_realism_brief"] = {
            "ok": qa_realism.get("ok", True),
            "summary": qa_realism.get("summary", ""),
            "failed": qa_realism.get("failed", []),
        }

        # Pour l'afficheur du runner : une vue compacte
        ctx.artifacts["qa_checklist"] = {
            "checks": qa_realism.get("checks", {}),
            "metrics": qa_realism.get("metrics", {}),
        }

        # --- Pretty checklist with emojis for runner display ---
        try:
            checks_disp = qa_realism.get("checks", {}) or {}
            metrics_disp = qa_realism.get("metrics", {}) or {}

            def _mark(x: bool) -> str:
                return "✅" if bool(x) else "❌"

            rows = [
                f"{_mark(checks_disp.get('cadence_10hz', True))} 🕒 Fréquence 10Hz correcte",
                f"{_mark(checks_disp.get('start_zero', True))} 🚦 Vitesse départ nulle",
                f"{_mark(checks_disp.get('end_zero', True))} 🛑 Vitesse fin nulle",
                f"{_mark(checks_disp.get('speed_nonnegative', True))} 🚗 Vitesse non négative",
                f"{_mark(checks_disp.get('ax_variability', True))} 📉 Variations inertielle réalistes (acc_x)",
                f"{_mark(checks_disp.get('gz_variability', True))} 🌀 Variations gyroscopiques réalistes (gyro_z)",
                f"{_mark(checks_disp.get('lateral_consistency', True))} 📊 Cohérence latérale (ay ≈ v·gz)",
                f"{_mark(checks_disp.get('turn_radius_plausible', True))} 📐 Rayon de virage plausible",
            ]

            # Compact metrics block
            metrics_lines: list[str] = []

            def _fmt(label: str, key: str):
                val = metrics_disp.get(key, None)
                if val is None:
                    return None
                try:
                    if isinstance(val, (int, float, np.floating)) and not np.isnan(val):
                        return f" - {label}: {val:.4g}"
                except Exception:
                    pass
                return f" - {label}: {val}"

            for label, key in [
                ("dt_median_s", "dt_median_s"),
                ("hz_obs", "hz_obs"),
                ("v_median_mps", "v_median_mps"),
                ("lat_err_mps2_med", "lat_err_mps2_med"),
                ("turn_radius_gz_ratio", "turn_radius_from_gz_ratio"),
            ]:
                line = _fmt(label, key)
                if line:
                    metrics_lines.append(line)

            status = "✅ OK" if qa_realism.get("ok", True) else f"❌ KO — {qa_realism.get('summary', '')}"
            pretty = "\n".join(rows + (["[Metrics]"] + metrics_lines if metrics_lines else []))

            # Expose a ready-to-print artifact for the runner
            ctx.artifacts["qa_pretty"] = {
                "status": status,
                "text": pretty,
            }
        except Exception:
            # Non-blocking: emoji rendering is best-effort
            pass

        # Politique de fail contrôlée par config
        if fail_on_realism and not qa_realism.get("ok", True):
            return Result(ok=False, message=f"Realism check failed: {qa_realism.get('summary', 'KO')}")

        return Result()
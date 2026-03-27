# engine/stages/validators.py
from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd
import numpy as np

from rs3_contracts.api import Result
from ..context import Context

logger = logging.getLogger(__name__)


def _compute_checklist(df: pd.DataFrame, hz_target: float = 10.0) -> tuple[dict, dict]:
    """Calcule une checklist standardisée (✅/❌) et quelques métriques.
    Retourne (checks, metrics).
    """
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}

    # --- Cadence & timeline ---
    t = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    # ns depuis epoch (astype('int64') ≈ ns, évite .view deprecated)
    ns = t.astype("int64").to_numpy()
    dt = np.diff((ns - ns[0]) / 1e9, prepend=0.0)
    pos = dt > 0
    if not pos.any():
        checks["cadence_10hz"] = False
        metrics["dt_median_s"] = float("nan")
        metrics["hz_obs"] = float("nan")
        dt_med = float("nan")
    else:
        dt_med = float(np.median(dt[pos]))
        metrics["dt_median_s"] = dt_med
        metrics["hz_obs"] = (1.0 / dt_med) if dt_med > 0 else float("nan")
        # tolérance ±20 ms à 10 Hz
        checks["cadence_10hz"] = abs(dt_med - 1.0 / hz_target) < 0.02

    # --- Vitesse (PRIORITÉ à 'speed' en m/s ; 'speed_kmh' juste en secours) ---
    sp_mps = pd.to_numeric(df.get("speed", 0.0), errors="coerce").fillna(0.0).to_numpy()
    if (np.asarray(sp_mps) == 0.0).all() and "speed_kmh" in df.columns:
        # fallback si speed vide
        sp_kmh = pd.to_numeric(df["speed_kmh"], errors="coerce").fillna(0.0).to_numpy()
        sp_mps = sp_kmh / 3.6
    checks["speed_nonnegative"] = bool((sp_mps >= -1e-6).all())

    # --- Départ/fin à 0 (médiane sur 1 s) ---
    n_tail = max(1, int(round(hz_target * 1.0)))  # 1 s window
    start0 = float(np.nanmedian(sp_mps[:n_tail])) < 0.6  # 0.6 m/s ≈ 2.16 km/h
    end0 = float(np.nanmedian(sp_mps[-n_tail:])) < 0.6
    checks["start_zero"] = start0
    checks["end_zero"] = end0

    # --- Variabilité inertielle minimale (sur mouvement) ---
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
    # Skip gz_variability si gyro désactivé (tout NaN ou tout 0)
    gyro_present = "gyro_z" in df.columns and not df["gyro_z"].isna().all() and not (df["gyro_z"].fillna(0) == 0).all()
    checks["gz_variability"] = std_gz > 0.002 if gyro_present else True  # pass si absent

    # --- Cohérence latérale sur virages (ay ≈ v * gz) + rayon plausible ---
    ay = pd.to_numeric(df.get("acc_y", 0.0), errors="coerce").fillna(0.0).to_numpy()
    turn = np.logical_and(moving, np.abs(gz) > 0.01)
    if turn.any():
        # 1) Cohérence latérale: ay ≈ v * gz
        ay_pred = sp_mps[turn] * gz[turn]
        err = float(np.nanmedian(np.abs(ay[turn] - ay_pred)))
        metrics["lat_err_mps2_med"] = err
        checks["lateral_consistency"] = err < 0.5  # tolérance m/s²

        # 2) Rayon de courbure plausible
        #   a) via gyro: R_gz ≈ v / |gz|
        R_gz = sp_mps[turn] / np.maximum(np.abs(gz[turn]), 1e-3)
        plausible_gz = np.logical_and(R_gz > 10.0, R_gz < 3000.0)
        ratio_gz = float(np.mean(plausible_gz)) if R_gz.size else 1.0
        metrics["turn_radius_from_gz_ratio"] = ratio_gz
        metrics["turn_radius_from_gz_med"] = float(np.nanmedian(R_gz)) if R_gz.size else float("nan")

        #   b) indicatif via ay: R_ay ≈ v² / |ay|
        R_ay = (sp_mps[turn] ** 2) / np.maximum(np.abs(ay[turn]), 1e-3)
        plausible_ay = np.logical_and(R_ay > 10.0, R_ay < 3000.0)
        ratio_ay = float(np.mean(plausible_ay)) if R_ay.size else 1.0
        metrics["turn_radius_from_ay_ratio"] = ratio_ay
        metrics["turn_radius_from_ay_med"] = float(np.nanmedian(R_ay)) if R_ay.size else float("nan")

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
    """Vérifications de réalisme légères, sans dépendre d'outils externes.
    Retourne un dict avec: available, ok, checks, summary, failed, metrics.
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
    Valide la cohérence du DataFrame final et exécute les vérifications de réalisme *intégrées*.
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
            return Result((False, "df vide"))

        vcfg: Dict[str, Any] = {}
        if isinstance(ctx.cfg, dict):
            vcfg = ctx.cfg.get("validation", {}) or {}

        fail_on_realism: bool = bool(vcfg.get("fail_on_realism", False))
        pass_basic: bool = bool(vcfg.get("pass_basic_checks", True))

        # -------------------------
        # 1) Checks de base (légers)
        # -------------------------
        if pass_basic:
            if "timestamp" not in df.columns:
                return Result((False, "timestamp manquant"))
            ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            if ts.isna().any():
                n_bad = int(ts.isna().sum())
                logger.warning("Timestamps invalides détectés: %d", n_bad)
                if vcfg.get("fail_on_nan", False):
                    return Result((False, f"{n_bad} timestamps invalides"))
            if not ts.is_monotonic_increasing:
                return Result((False, "timestamps non monotones"))
            if df.isnull().any().any():
                logger.warning("NaN détectés dans le DataFrame de sortie.")
                ctx.artifacts["qa_basic"] = {"nan_detected": True}
                if vcfg.get("fail_on_nan", False):
                    return Result((False, "NaN détectés"))
            else:
                ctx.artifacts["qa_basic"] = {"nan_detected": False}

        # --- Garantit les colonnes IMU numériques ---
        # acc : toujours requis (fillna 0)
        # gyro : respecter les NaN (gyro désactivé par MultiRateSampler)
        df = df.copy()
        for col in ["acc_x", "acc_y", "acc_z"]:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        for col in ["gyro_x", "gyro_y", "gyro_z"]:
            if col not in df.columns:
                df[col] = 0.0
            elif not df[col].isna().all():
                # Gyro actif : fillna 0 pour les valeurs manquantes ponctuelles
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            # Si tout NaN (gyro désactivé) : on laisse NaN

        # --- Colonne event si absente ---
        if "event" not in df.columns:
            df["event"] = ""

        # Propagation vers le contexte (Exporter fera le cast/schema)
        ctx.df = df

        # -------------------------
        # 2) Check réalisme intégré
        # -------------------------
        try:
            qa_realism: Dict[str, Any] = _realism_lite(df, hz_target=float(ctx.meta.get("hz", 10)))
        except Exception as e:
            logger.exception("realism_lite a échoué: %s", e)
            qa_realism = {"available": True, "ok": False, "error": str(e), "summary": "KO"}

        # --- Ajout checklist standardisée pour affichage (✅/❌) ---
        try:
            std_checks, std_metrics = _compute_checklist(df, hz_target=float(ctx.meta.get("hz", 10)))
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
            failed = d.get("failed")
            if isinstance(failed, (list, tuple)) and failed:
                return ", ".join(map(str, failed[:5]))
            checks = d.get("checks")
            if isinstance(checks, dict):
                bad = [k for k, v in checks.items() if v is False]
                if bad:
                    return ", ".join(bad[:5])
            for key in ("errors", "warnings", "issues", "messages"):
                val = d.get(key)
                if isinstance(val, (list, tuple)) and val:
                    return ", ".join(map(str, val[:3]))
                if isinstance(val, str) and val:
                    return val
            return "raison inconnue"

        # Résumé & statut
        if "ok" not in qa_realism:
            failed_from_checks = [k for k, v in (qa_realism.get("checks") or {}).items() if v is False]
            qa_realism["failed"] = failed_from_checks
            qa_realism["ok"] = len(failed_from_checks) == 0

        if qa_realism.get("ok", True):
            summary = "OK"
            status_level = "ok"
        else:
            reason = _summarize_failure(qa_realism)
            if reason == "raison inconnue":
                bad = [k for k, v in (qa_realism.get("checks") or {}).items() if v is False]
                if bad:
                    reason = ", ".join(bad[:5])
            summary = f"KO: {reason}"
            status_level = "ko"
            logger.warning("Realism KO — causes: %s", reason)
        qa_realism["summary"] = summary
        qa_realism["status_level"] = status_level  # utile au template HTML

        ctx.artifacts["qa_realism"] = qa_realism
        ctx.artifacts["qa_realism_brief"] = {
            "ok": qa_realism.get("ok", True),
            "summary": qa_realism.get("summary", ""),
            "failed": qa_realism.get("failed", []),
            "status_level": qa_realism.get("status_level", "ok"),
        }

        # Pour l'afficheur du runner : vue compacte
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

            ctx.artifacts["qa_pretty"] = {
                "status": status,
                "text": pretty,
            }
        except Exception:
            # Non-bloquant : rendu émojis best-effort
            pass

        if fail_on_realism and not qa_realism.get("ok", True):
            return Result((False, f"Realism check failed: {qa_realism.get('summary', 'KO')}"))

        return Result((True, "OK"))
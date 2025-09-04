# runner/run_simulation2.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import logging
from typing import List, Any

from core2.pipeline import PipelineSimulator
from core2.context import Context

from core2.stages.legs_plan import LegsPlan
from core2.stages.legs_route import LegsRoute
from core2.stages.legs_stitch import LegsStitch
from core2.stages.stopwait_injector import StopWaitInjector
from core2.stages.stop_smoother import StopSmoother
from core2.stages.imu_projector import IMUProjector
from core2.stages.noise_injector import NoiseInjector
from core2.stages.events_tagger import EventsTagger
from core2.stages.validators import Validators
from core2.stages.exporter import Exporter
from core2.stages.road_enricher import RoadEnricher
from core2.stages.speed_limiter import SpeedLimiter
from core2.stages.speed_smoother import SpeedSmoother
from core2.stages.speed_sync import SpeedSync
from core2.stages.final_stop_locker import FinalStopLocker
from core2.stages.geo_spike_filter import GeoSpikeFilter
from core2.stages.legs_retimer import LegsRetimer
from core2.plugin_loader import discover_external_stages
from core2.stages.initial_stop_locker import InitialStopLocker


logger = logging.getLogger(__name__)


def _resolve_config_path(raw: str) -> Path:
    """
    Résout un chemin de config de manière robuste:
    - tel que passé (absolu ou relatif au CWD)
    - relatif au dossier du runner (../config)
    - ou par simple nom de fichier sous ../config
    """
    if not raw:
        raw = "config/simulator.yaml"
    p = Path(raw)
    if p.exists():
        return p

    runner_dir = Path(__file__).resolve().parent
    proj_root = runner_dir.parent

    candidates = [
        runner_dir / raw,
        proj_root / raw,
        proj_root / "config" / Path(raw).name,
        proj_root / "config" / "simulator.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Config introuvable. Essais: {[str(x) for x in [p, *candidates]]}")


def build_pipeline(cfg):
    osrm_cfg = cfg.get("osrm", {}) if isinstance(cfg, dict) else {}
    profile = osrm_cfg.get("profile", "driving")
    max_workers = int(osrm_cfg.get("max_workers", 4))

    stop_smoother_cfg = cfg.get("stop_smoother", {}) if isinstance(cfg, dict) else {}
    events_tagger_cfg = cfg.get("events_tagger", {}) if isinstance(cfg, dict) else {}
    noise_injector_cfg = cfg.get("noise_injector", {}) if isinstance(cfg, dict) else {}
    speed_limiter_cfg = cfg.get("speed_limiter", {}) if isinstance(cfg, dict) else {}
    speed_smoother_cfg = cfg.get("speed_smoother", {}) if isinstance(cfg, dict) else {}
    final_stop_locker_cfg = cfg.get("final_stop_locker", {}) if isinstance(cfg, dict) else {}
    geo_spike_filter_cfg = cfg.get("geo_spike_filter", {}) if isinstance(cfg, dict) else {}
    legs_retimer_cfg = cfg.get("legs_retimer", {}) if isinstance(cfg, dict) else {}
    speed_sync_cfg = cfg.get("speed_sync", {}) if isinstance(cfg, dict) else {}

    extras = discover_external_stages(cfg)

    # Log plugin stages discovered so we clearly see they will run
    if extras:
        try:
            names = ", ".join(getattr(s, "name", type(s).__name__) for s in extras)
        except Exception:
            names = ", ".join(type(s).__name__ for s in extras)
        print(f"[Plugins] Stages découverts: {names}")
    else:
        print("[Plugins] Aucun stage externe détecté.")

    def _stage_name(obj: Any) -> str:
        """Nom lisible d'un stage (priorité à l'attribut .name)."""
        try:
            n = getattr(obj, "name", None)
            if isinstance(n, str) and n:
                return n
        except Exception:
            pass
        return type(obj).__name__

    def _find_index_by_class(stages: List[Any], class_name: str) -> int | None:
        """Retourne l'index du premier stage dont la classe se nomme class_name."""
        for i, st in enumerate(stages):
            if type(st).__name__ == class_name:
                return i
        return None

    def _default_anchor_index(stages: List[Any]) -> int:
        """Position par défaut d'insertion des plugins (après SpeedSync si présent, sinon avant IMUProjector, sinon fin)."""
        i = _find_index_by_class(stages, "SpeedSync")
        if i is not None:
            return i + 1
        j = _find_index_by_class(stages, "IMUProjector")
        if j is not None:
            return j
        return len(stages)

    def _place_plugins(stages: List[Any], plugins: List[Any], cfg: dict) -> List[Any]:
        """
        Insère les stages plugins selon la config:
        plugins:
          insert:
            - name: AltitudeStage         # .name OU nom de classe
              before: IMUProjector        # (optionnel) ancre par nom de classe
            - name: MyOtherPlugin
              after: SpeedSync
        Les entrées sont traitées dans l'ordre. Les plugins non référencés sont insérés à l'ancre par défaut.
        """
        rules = (cfg.get("plugins", {}) or {}).get("insert", []) if isinstance(cfg, dict) else []
        remaining = plugins.copy()

        def _pop_plugin_by_name(wanted: str) -> Any | None:
            for k, p in enumerate(remaining):
                if _stage_name(p) == wanted or type(p).__name__ == wanted:
                    return remaining.pop(k)
            return None

        # Applique les règles explicites
        for r in rules:
            try:
                pname = str(r.get("name", "")).strip()
                if not pname:
                    continue
                p = _pop_plugin_by_name(pname)
                if p is None:
                    logger.warning("[Plugins] Règle ignorée: plugin '%s' introuvable.", pname)
                    continue

                idx = None
                if "before" in r and r["before"]:
                    idx_target = _find_index_by_class(stages, str(r["before"]))
                    if idx_target is not None:
                        idx = idx_target
                if idx is None and "after" in r and r["after"]:
                    idx_target = _find_index_by_class(stages, str(r["after"]))
                    if idx_target is not None:
                        idx = idx_target + 1

                if idx is None:
                    idx = _default_anchor_index(stages)

                stages.insert(idx, p)
                logger.info("[Plugins] '%s' inséré à l'index %d.", _stage_name(p), idx)
            except Exception as e:
                logger.warning("[Plugins] Échec d'insertion d'un plugin (%s): %s", r, e)

        # Place les plugins restants à l'ancre par défaut
        if remaining:
            idx = _default_anchor_index(stages)
            for p in remaining:
                stages.insert(idx, p)
                idx += 1
                logger.info("[Plugins] '%s' inséré (par défaut) à l'index %d.", _stage_name(p), idx-1)

        return stages

    # Construit SpeedSync avec la config si possible, fallback sinon
    try:
        speed_sync_stage = SpeedSync(**({} | speed_sync_cfg))
    except Exception as e:
        logger.warning("[SpeedSync] Paramètres invalides dans la config (%s) — utilisation des défauts.", e)
        speed_sync_stage = SpeedSync()

    stages: List[Any] = [
        LegsPlan(),
        LegsRoute(profile=profile, max_workers=max_workers),
        LegsStitch(),
        GeoSpikeFilter(**({
            "vmax_kmh": 120.0,       # plafond réaliste pour réseau secondaire/urbain
            "hard_jump_m": 60.0,     # cap dur sur le bond inter-échantillons
            "soft_margin_m": 2.0     # marge douce adaptée au 10 Hz (≈20 cm par 100 ms)
        } | geo_spike_filter_cfg)),
        RoadEnricher(),
        LegsRetimer(**({
            "default_kmh": 50.0,
            "use_column_target_speed": True,
            "min_dt": 0.05
        } | legs_retimer_cfg)),
        SpeedLimiter(**({
            "source_unit": "kmh",
            "margin_kmh": 3.0,
            "ramp_s": 0.0
        } | speed_limiter_cfg)),
        StopWaitInjector(),
        StopSmoother(**({
            "v_in": 0.25, "t_in": 2.0,
            "v_out": 0.6, "t_out": 2.5,
            "lock_pos": True
        } | stop_smoother_cfg)),
        SpeedSmoother(**({
            "window_s": 1.5,
            "min_periods": 1
        } | speed_smoother_cfg)),
        InitialStopLocker(**(cfg.get("initial_stop_locker", {}) or {})),
        FinalStopLocker(**({
            "tail_s": 8.0
        } | final_stop_locker_cfg)),
        speed_sync_stage,
        IMUProjector(),
        NoiseInjector(**({
            "sigma_acc": 0.02,
            "sigma_gyro": 0.001
        } | noise_injector_cfg)),
        EventsTagger(**({
            "dvdt_thr_mps2": 0.5,
            "head_window_s": 10.0,
            "tail_window_s": 10.0
        } | events_tagger_cfg)),
        Validators(),
        Exporter(),
    ]

    # Injection des plugins selon la config
    stages = _place_plugins(stages, extras, cfg)

    return PipelineSimulator(stages)


def main():
    ap = argparse.ArgumentParser(description="RoadSimulator3 — pipeline legs->WAIT/STOP")
    ap.add_argument("--config", default="config/simulator.yaml", help="Chemin du YAML de simulation")
    ap.add_argument("--schema", default=None, help="Chemin du schéma dataset (optionnel)")

    args = ap.parse_args()

    cfg_path = _resolve_config_path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Permet d'overrider le chemin du schéma dataset sans toucher au YAML
    if args.schema:
        cfg = dict(cfg)
        cfg["dataset_schema"] = str(args.schema)

    ctx = Context(cfg=cfg)
    pipeline = build_pipeline(cfg)
    pipeline.run(ctx)

    outdir = ctx.meta.get("outdir", cfg.get("outdir", "data/simulations"))
    print(f"[OK] Pipeline terminé. Sortie → {outdir}")

    # Petit résumé QA si disponible (sans refaire de calcul, pas d'impact Hz)
    qa = ctx.artifacts.get("qa_realism")
    if isinstance(qa, dict):
        summary = qa.get("summary", "n/a")
        ok = qa.get("ok", True)
        print(f"[QA] Realism: {'OK' if ok else 'KO'} — {summary}")

    # Affichage enrichi (checklist emoji) si dispo
    pretty = ctx.artifacts.get("qa_pretty")
    if isinstance(pretty, dict):
        status = pretty.get("status")
        text = pretty.get("text")
        if status:
            print(status)
        if text:
            print(text)


if __name__ == "__main__":
    main()
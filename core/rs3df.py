from __future__ import annotations
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

import os
import logging
import json

logger = logging.getLogger(__name__)

# --- Règles & défauts ---------------------------------------------------------

@dataclass(frozen=True)
class EventRule:
    cooldown_s: float = 0.8   # anti re-déclenchement temporel
    min_gap_m: float = 5.0    # distance mini entre 2 mêmes events
    priority: int = 50        # réservé si besoin (non utilisé ici)

DEFAULT_RULES: Dict[str, EventRule] = {
    "brake":       EventRule(cooldown_s=0.8,  min_gap_m=8.0,  priority=10),
    "accel":       EventRule(cooldown_s=0.8,  min_gap_m=8.0,  priority=20),
    "speed_bump":  EventRule(cooldown_s=1.0,  min_gap_m=12.0, priority=5),
    "pothole":     EventRule(cooldown_s=0.7,  min_gap_m=10.0, priority=15),
    "curb":        EventRule(cooldown_s=0.7,  min_gap_m=10.0, priority=15),
}

# --- Utilitaires internes -----------------------------------------------------

def _round_time_s(t: float) -> float:
    """Ancrage à 0.1 s (cadence 10 Hz)."""
    return float(np.round(t, 1))


def _round_coord(x: float) -> float:
    return float(np.round(x, 6))


class RS3DF:
    """Squelette minimal du conteneur DataFrame + API d'événements.

    Exigences (aligné tests):
    - colonnes requises: timestamp, lat, lon
    - colonnes événements auto-ajoutées si absentes: event, event_id, event_type, idx_anchor, t_anchor
    - add_event_* crée 1 ancrage unique sur la ligne pivot (10 Hz) et retourne un event_id (str)
    - dédoublonnage strict par clé (event_type, t_anchor(0.1s), lat6, lon6)
    - garde-fou cooldown/min_gap par type d'événement
    - remove_event supprime l'ancrage (retourne bool)
    - quality_asserts: absence de doublons (t,lat6,lon6) et unicité de chaque event_id
    - to_csv/to_json valident avant export
    """

    def __init__(self, df: pd.DataFrame, rules: Optional[Dict[str, EventRule]] = None):
        self.df = df.copy()
        self.rules = rules or DEFAULT_RULES
        # registre anti-doublons clé exacte & dernier par type
        self._seen: set[Tuple[str, float, float, float]] = set()
        self._last_by_type: Dict[str, Dict[str, Any]] = {}

        required = {"timestamp", "lat", "lon"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")

        # Crée colonnes événements si absentes
        for col, dtype in [
            ("event", "object"),
            ("event_id", "object"),
            ("event_type", "object"),
            ("idx_anchor", "float"),
            ("t_anchor", "float"),
        ]:
            if col not in self.df.columns:
                self.df[col] = np.nan
            if dtype == "object":
                self.df[col] = self.df[col].astype("object")

    # --- helpers --------------------------------------------------------------

    def _row(self, idx: int) -> pd.Series:
        return self.df.iloc[idx]

    def _haversine_m(self, lat1, lon1, lat2, lon2) -> float:
        r = 6371000.0
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlmb = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
        return float(2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

    def _too_close_last(self, event_type: str, t_anchor: float, lat: float, lon: float) -> bool:
        last = self._last_by_type.get(event_type)
        if not last:
            return False
        rule = self.rules.get(event_type, EventRule())
        if abs(t_anchor - last["t_anchor"]) < rule.cooldown_s:
            return True
        return self._haversine_m(lat, lon, last["lat"], last["lon"]) < rule.min_gap_m

    # --- API publique ---------------------------------------------------------

    def add_event_at_idx(self, event_type: str, idx_anchor: int, payload: Optional[Dict[str, Any]] = None) -> Optional[str]:
        if event_type not in self.rules:
            raise ValueError(f"Type d'événement inconnu: {event_type}")

        row = self._row(idx_anchor)
        t_anchor = _round_time_s(float(row["timestamp"]))
        latr = _round_coord(float(row["lat"]))
        lonr = _round_coord(float(row["lon"]))
        key = (event_type, t_anchor, latr, lonr)

        # 1) dédup clé exacte
        if key in self._seen:
            return None
        # 2) cooldown / min-gap
        if self._too_close_last(event_type, t_anchor, latr, lonr):
            return None

        # Ancrage unique sur la ligne pivot
        eid = str(uuid.uuid4())[:8]
        self.df.loc[self.df.index[idx_anchor], [
            "event", "event_id", "event_type", "idx_anchor", "t_anchor"
        ]] = [event_type, eid, event_type, idx_anchor, t_anchor]

        self._seen.add(key)
        self._last_by_type[event_type] = {
            "t_anchor": t_anchor, "lat": latr, "lon": lonr, "idx_anchor": int(idx_anchor)
        }
        return eid

    def add_event_at_time(self, event_type: str, t_seconds: float) -> Optional[str]:
        t_anchor = _round_time_s(float(t_seconds))
        idx = int(np.argmin(np.abs(self.df["timestamp"].values - t_anchor)))
        return self.add_event_at_idx(event_type, idx)

    def add_event_at_geo(self, event_type: str, lat: float, lon: float) -> Optional[str]:
        latr, lonr = _round_coord(float(lat)), _round_coord(float(lon))
        d = self._haversine_m(latr, lonr, self.df["lat"].values, self.df["lon"].values)
        idx = int(np.argmin(d))
        return self.add_event_at_idx(event_type, idx)

    def remove_event(self, event_id: str) -> bool:
        sel = self.df["event_id"] == event_id
        if not sel.any():
            return False
        self.df.loc[sel, ["event", "event_id", "event_type", "idx_anchor", "t_anchor"]] = [np.nan] * 5
        return True

    # --- Qualité & export -----------------------------------------------------

    def quality_asserts(self) -> None:
        # Pas de doublons (t, lat6, lon6)
        lat6 = self.df["lat"].round(6)
        lon6 = self.df["lon"].round(6)
        t01 = self.df["timestamp"].round(1)
        dup = pd.Series(list(zip(t01, lat6, lon6))).duplicated().any()
        assert not dup, "Doublons (timestamp,lat6,lon6) détectés"

        # Un et un seul ancrage par event_id
        tmp = self.df.dropna(subset=["event_id"])
        if len(tmp):
            counts = tmp["event_id"].value_counts()
            assert (counts == 1).all(), "Un event_id est posé sur >1 ligne (ancrage non-unique)."

    def to_csv(self, path: str) -> None:
        self.quality_asserts()
        self.df.to_csv(path, index=False)

    def to_json(self, path: str, orient: str = "records") -> None:
        self.quality_asserts()
        self.df.to_json(path, orient=orient)


# ---------------------------------------------------------------------
# RS3DS: couche dataset I/O + conventions pour RoadSimulator3
# ---------------------------------------------------------------------
from typing import Optional, Dict, Any

class RS3DS:
    """Couche *dataset* (I/O + conventions) pour RoadSimulator3.

    - Crée un dossier de sortie standardisé : data/simulations/simulated_<timestamp>/
    - Fournit des helpers d'export (CSV/JSON) et un symlink vers la dernière trace
    - Optionnellement, applique des validations légères de schéma

    Exemple d'usage (depuis runner.run_simulation) :
    >>> ds = RS3DS(timestamp, config)
    >>> out = ds.output_dir
    >>> ds.symlink_last_trace(os.path.join(out, "trace.csv"))
    """

    # Ordre strict de référence v1.0 (si colonnes présentes)
    STRICT_ORDER = (
        "timestamp", "lat", "lon", "speed",
        "acc_x", "acc_y", "acc_z", "event",
    )
    # Casting conseillé pour export (si colonnes présentes)
    DTYPE_MAP = {
        "timestamp": float,  # secondes (avec ms) sans timezone
        "lat": float,
        "lon": float,
        "speed": float,
        "acc_x": float,
        "acc_y": float,
        "acc_z": float,
        "event": object,    # NaN si vide
    }

    REQUIRED_COLUMNS = ("timestamp", "lat", "lon")

    def __init__(self, timestamp: str, config: Optional[Dict[str, Any]] = None,
                 base_dir: str = os.path.join("data", "simulations")) -> None:
        self.timestamp = timestamp
        self.config = config or {}
        self.base_dir = base_dir
        self.output_dir = os.path.join(base_dir, f"simulated_{timestamp}")
        self._ensure_dirs()
        logger.debug("[RS3DS] Output dir ready: %s", self.output_dir)

    # ----------------------------
    # Dossiers / chemins
    # ----------------------------
    def _ensure_dirs(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def path(self, *parts: str) -> str:
        """Construit un chemin sous output_dir."""
        return os.path.join(self.output_dir, *parts)

    # ----------------------------
    # Manifest & méta
    # ----------------------------
    def write_manifest(self, df: Optional[pd.DataFrame] = None, extra: Optional[Dict[str, Any]] = None) -> str:
        """Écrit un petit manifest JSON dans le dossier de sortie.
        Contenu minimal : timestamp, nb_points (si df fourni), version (si trouvable),
        et un échantillon de colonnes.
        """
        meta: Dict[str, Any] = {
            "timestamp": self.timestamp,
            "output_dir": self.output_dir,
        }
        # tentative de version depuis config
        version = None
        try:
            version = (self.config or {}).get("version")
        except Exception:
            version = None
        if version is None and isinstance(self.config, dict):
            # fallback : chemin VERSION dans repo non garanti ici
            version = self.config.get("app_version")
        if version is not None:
            meta["version"] = version

        if df is not None:
            try:
                meta["nb_points"] = int(len(df))
                meta["columns"] = list(df.columns)
            except Exception:
                pass
        if extra:
            meta.update(extra)

        manifest_path = self.path("metadata.json")
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            logger.info("[RS3DS] Manifest écrit : %s", manifest_path)
        except OSError as e:
            logger.warning("[RS3DS] Impossible d'écrire le manifest (%s)", e)
        return manifest_path

    # ----------------------------
    # Schéma & validations
    # ----------------------------
    @classmethod
    def validate_min_schema(cls, df: pd.DataFrame) -> None:
        missing = set(cls.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes obligatoires manquantes: {missing}")

    @classmethod
    def enforce_export_schema(cls, df: pd.DataFrame, strict_order: bool = True, cast_dtypes: bool = True) -> pd.DataFrame:
        """Réordonne et *cast* les colonnes connues selon la spec v1.0 si présentes.
        - `strict_order` : place STRICT_ORDER en tête
        - `cast_dtypes`  : applique DTYPE_MAP quand possible
        """
        out = df.copy()
        if cast_dtypes:
            for col, dt in cls.DTYPE_MAP.items():
                if col in out.columns:
                    try:
                        out[col] = out[col].astype(dt)
                    except Exception:
                        # on n'échoue pas l'export pour un cast non critique
                        logger.debug("[RS3DS] Cast ignoré pour %s", col)
        if strict_order:
            head = [c for c in cls.STRICT_ORDER if c in out.columns]
            tail = [c for c in out.columns if c not in head]
            out = out[head + tail]
        return out

    # ----------------------------
    # Exports
    # ----------------------------
    def export_csv(self, df: pd.DataFrame, filename: str = "trace.csv",
                   validate_schema: bool = True, use_rs3df_quality: bool = True) -> str:
        """Exporte un CSV dans le dossier de la simulation.

        - `validate_schema` : vérifie les colonnes minimales
        - `use_rs3df_quality` : applique les *quality_asserts* de RS3DF avant export
        """
        if validate_schema:
            RS3DS.validate_min_schema(df)
        out_df = RS3DS.enforce_export_schema(df, strict_order=True, cast_dtypes=True)
        out_path = self.path(filename)
        if use_rs3df_quality:
            RS3DF(out_df).to_csv(out_path)
        else:
            out_df.to_csv(out_path, index=False)
        logger.info("[RS3DS] CSV exporté : %s", out_path)
        # manifest auto
        try:
            self.write_manifest(out_df, extra={"export": os.path.basename(filename)})
        except Exception:
            pass
        return out_path

    def export_json(self, df: pd.DataFrame, filename: str = "trace.json",
                    orient: str = "records", use_rs3df_quality: bool = True) -> str:
        if RS3DS.validate_min_schema:
            RS3DS.validate_min_schema(df)
        out_df = RS3DS.enforce_export_schema(df, strict_order=True, cast_dtypes=True)
        out_path = self.path(filename)
        if use_rs3df_quality:
            RS3DF(out_df).to_json(out_path, orient=orient)
        else:
            out_df.to_json(out_path, orient=orient)
        logger.info("[RS3DS] JSON exporté : %s", out_path)
        try:
            self.write_manifest(out_df, extra={"export": os.path.basename(filename)})
        except Exception:
            pass
        return out_path

    # ----------------------------
    # Symlink vers la dernière trace
    # ----------------------------
    def symlink_last_trace(self, trace_path: str,
                           link_name: str = os.path.join("data", "simulations", "last_trace.csv")) -> str:
        """Crée/rafraîchit un lien symbolique vers la dernière trace générée."""
        try:
            if os.path.islink(link_name) or os.path.exists(link_name):
                os.remove(link_name)
            os.symlink(trace_path, link_name)
            logger.info("[RS3DS] Lien symbolique créé : %s → %s", link_name, trace_path)
        except OSError as e:
            logger.warning("[RS3DS] Impossible de créer le lien symbolique (%s)", e)
        return link_name

    # ----------------------------
    # Intégration avec RS3DF (facultatif)
    # ----------------------------
    def wrap(self, df: pd.DataFrame, rules: Optional[Dict[str, EventRule]] = None) -> RS3DF:
        """Retourne un conteneur RS3DF prêt pour l'édition/l'ancrage d'événements."""
        RS3DS.validate_min_schema(df)
        return RS3DF(df, rules=rules)

    # ----------------------------
    # Exports standardisés (raccourci)
    # ----------------------------
    def export_standard_trace(self, df: pd.DataFrame, *, filename: str = "trace.csv",
                              create_symlink: bool = True) -> Tuple[str, Optional[str]]:
        """Exporte la trace CSV standard + symlink optionnel.
        Retourne (csv_path, symlink_path | None)
        """
        csv_path = self.export_csv(df, filename)
        link = None
        if create_symlink:
            link = self.symlink_last_trace(csv_path)
        return csv_path, link
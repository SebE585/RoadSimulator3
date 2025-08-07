import logging
import numpy as np
import pandas as pd

from simulator.events.tracker import EventCounter

from core.utils import ensure_strictly_increasing_timestamps
from simulator.events.generation import (
    generate_opening_door,
    generate_dos_dane,
    generate_nid_de_poule,
    generate_trottoir,
    generate_freinage,
    generate_acceleration
)
from simulator.events.stop_wait import apply_progressive_acceleration_after_stop_wait
from simulator.events.utils import clean_invalid_events
from simulator.vizualisation.generate_map_comparison import TraceDebugger

logger = logging.getLogger(__name__)

def apply_postprocessing(df, hz=None, config=None):
    """
    Nettoyage post-pipeline des événements :
    - Supprime les événements invalides ou incohérents.
    - Corrige les valeurs extrêmes et réordonne les timestamps si nécessaire.

    Args:
        df (pd.DataFrame): Données simulées.
        hz (int, optional): Fréquence d'échantillonnage.
        config (dict, optional): Configuration optionnelle.

    Returns:
        pd.DataFrame: Données nettoyées.
    """
    return clean_invalid_events(df)


def inject_all_events(df, config):
    if df.attrs.get("events_already_injected", False):
        logger.warning("⚠️ Les événements ont déjà été injectés dans ce DataFrame. Injection ignorée pour éviter les doublons.")
        return df
    df.attrs["events_already_injected"] = True
    """
    Injecte les événements inertiels (accélération, freinage, dos d’âne, etc.) 
    et applique l’accélération progressive après un arrêt. 
    Seules les ouvertures de porte sont conditionnelles dans ce mode minimal.

    Args:
        df (pd.DataFrame): Données interpolées à enrichir.
        config (dict): Configuration des événements.

    Returns:
        pd.DataFrame: Données enrichies avec événements simulés.
    """
    logger.info("➡️ Injection minimale des événements (stop, wait, ouverture_porte)")
    tracker = EventCounter()

    debugger = TraceDebugger()
    debugger.save(df, label="01_before_injection")

    # Suppression du premier 'stop' s'il est en tête de trajectoire
    if df.iloc[0]["event"] == "stop":
        logger.info("🚦 Suppression du premier 'stop' considéré comme point de départ.")
        df.loc[0, "event"] = np.nan

    def maybe_inject(name, func):
        """Injecte un événement si activé dans la configuration."""
        if not config.get("injection", {}).get(name, {}).get("enabled", True):
            return df
        df_before = df.copy()
        if name == "ouverture_porte":
            df_after = func(df, config)
        else:
            df_after = func(df, config=config)
        n_added = (
            df_after["event"].fillna("").str.contains(name).sum()
            - df_before["event"].fillna("").str.contains(name).sum()
        )
        tracker.add(name, n_added)
        debugger.save(df_after, label=f"03_after_{name}")
        return df_after

    # Injection de l'accélération progressive après les événements stop/wait
    df = apply_progressive_acceleration_after_stop_wait(
        df, hz=10, target_speed_kmh=30, duration_s=5
    )
    debugger.save(df, label="02_after_progressive_acceleration")

    # Événement "ouverture_porte" optionnel
    if config.get("injection", {}).get("ouverture_porte", {}).get("enabled", False):
        df = maybe_inject("ouverture_porte", generate_opening_door)

    # Événements inertiels standards
    for name, func in [
        ("acceleration", generate_acceleration),
        ("freinage", generate_freinage),
        ("dos_dane", generate_dos_dane),
        ("nid_de_poule", generate_nid_de_poule),
        ("trottoir", generate_trottoir),
    ]:
        df = maybe_inject(name, func)

    logger.info("📊 Événements injectés : " + tracker.summary())

    # Réordonnancement final
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = ensure_strictly_increasing_timestamps(df)
    debugger.save(df, label="04_final_after_sorting")

    df["event"] = df["event"].where(~df.duplicated(subset=["timestamp", "event"]), np.nan)

    df = deduplicate_event_labels(df)
    # Renforcer le nettoyage : appel une seconde fois pour supprimer les répétitions espacées par un NaN
    df = deduplicate_event_labels(df)
    # 🔒 Sécurité finale : garantir que les événements ponctuels sont uniques par position
    ponctuels = ["acceleration", "freinage", "dos_dane", "trottoir", "nid_de_poule", "ouverture_porte"]
    for event_name in ponctuels:
        idx = df.index[df["event"] == event_name].tolist()
        for i in range(1, len(idx)):
            if idx[i] == idx[i - 1] + 1:
                df.at[idx[i], "event"] = np.nan
    return df

def deduplicate_event_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les étiquettes d’événements répétées sur des lignes consécutives
    ou espacées par un seul NaN. Ex : [freinage, freinage] ou [freinage, NaN, freinage].
    """
    event_col = df["event"].fillna("")
    for i in range(2, len(df)):
        if event_col[i] == event_col[i - 2] and event_col[i - 1] == "":
            df.at[i, "event"] = np.nan
    is_same = (df["event"] == df["event"].shift()) & (~df["event"].isna())
    df.loc[is_same, "event"] = np.nan
    return df

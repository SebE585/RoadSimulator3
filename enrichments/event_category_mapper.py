

import logging
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

def project_event_categories(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """
    Ajoute des colonnes binaires par catégorie d'événements (event_infra, event_behavior, event_context).
    Conserve la colonne 'event' pour la rétro-compatibilité.

    Règles:
      - Lit le mapping event→category depuis config["dataset"]["events"]["categories"]
        ou depuis le fichier YAML event_categories.yaml (si schema_path fourni).
      - Pour chaque catégorie définie, crée une colonne 'event_<cat>' avec 0/1.
    """
    if df is None or df.empty:
        return df

    mapping = {}
    categories = []

    # Charger depuis config si possible
    try:
        if config and "dataset" in config and "events" in config["dataset"]:
            cats = config["dataset"]["events"].get("categories", [])
            if isinstance(cats, dict) and "mapping" in cats:
                mapping = cats["mapping"]
                categories = cats.get("categories", [])
    except Exception:
        logger.debug("No categories mapping found in config")

    # Sinon charger depuis event_categories.yaml
    if not mapping and config and "schema_path" in config:
        import os
        schema_dir = os.path.dirname(config["schema_path"])
        yaml_path = os.path.join(schema_dir, "event_categories.yaml")
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            mapping = data.get("mapping", {})
            categories = data.get("categories", [])
        except FileNotFoundError:
            logger.warning("event_categories.yaml introuvable")
        except Exception as e:
            logger.debug("Impossible de charger event_categories.yaml (%s)", e)

    if not categories:
        categories = sorted(set(mapping.values()))

    # Créer colonnes binaires par catégorie
    for cat in categories:
        col = f"event_{cat}"
        df[col] = 0

    if "event" not in df.columns:
        df["event"] = pd.Series([None] * len(df))

    for i, ev in df["event"].items():
        if not isinstance(ev, str):
            continue
        cat = mapping.get(ev)
        if cat:
            col = f"event_{cat}"
            if col in df.columns:
                df.at[i, col] = 1

    logger.debug("Event categories projected → %s", categories)
    return df
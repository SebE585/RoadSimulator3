"""
Chargement centralisé de la configuration du module events
via le fichier YAML de configuration.
"""

import yaml
import os

# Résout le chemin absolu vers le fichier `config/events.yaml`
CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "config", "events.yaml")
)

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

def get_event_config(event_name: str, default: dict = None) -> dict:
    try:
        return CONFIG["events"][event_name]
    except KeyError:
        if default is not None:
            logging.warning(f"[CONFIG] Clé 'events.{event_name}' manquante, utilisation des valeurs par défaut.")
            return default
        raise KeyError(f"[CONFIG] Clé 'events.{event_name}' absente du fichier de configuration.")
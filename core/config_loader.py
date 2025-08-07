import os
import yaml

# Chemin du fichier de configuration principal
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
SIMULATION_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'simulation.yaml')

_cached_config = None
_cached_simulation_config = None

def load_config(config_path=None):
    """
    Charge le fichier de configuration principal.
    """
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    if config_path is None:
        config_path = CONFIG_PATH
    
    with open(config_path, 'r') as f:
        _cached_config = yaml.safe_load(f)
    
    return _cached_config

def load_simulation_config(config_path=None):
    """
    Charge la configuration spécifique à la simulation (simulation.yaml).
    """
    global _cached_simulation_config
    if _cached_simulation_config is not None:
        return _cached_simulation_config

    if config_path is None:
        config_path = SIMULATION_CONFIG_PATH
    
    with open(config_path, 'r') as f:
        _cached_simulation_config = yaml.safe_load(f)
    
    return _cached_simulation_config

# Fonctions spécifiques
def get_heading_smooth_window():
    cfg = load_config()
    return cfg.get('general', {}).get('heading_smooth_window', 41)

def get_speed_smooth_window():
    cfg = load_config()
    return cfg.get('general', {}).get('speed_smooth_window', 21)

def get_context_config():
    cfg = load_config()
    return cfg.get('context', {})

def get_context_sources():
    return get_context_config().get('sources', {})

def get_context_thresholds():
    return get_context_config().get('thresholds', {})

def resolve_mnt_path(config):
    mnt_data = config.get('mnt_data', {})
    folder = mnt_data.get('local_folder', '')
    filenames = mnt_data.get('filenames', [])

    if folder and filenames:
        mnt_path = os.path.join(folder, filenames[0])
        if os.path.exists(mnt_path):
            print(f"[INFO] MNT résolu depuis la configuration : {mnt_path}")
            return mnt_path
        else:
            print(f"[ERROR] Le fichier MNT spécifié est introuvable : {mnt_path}")
    else:
        print("[ERROR] Informations mnt_data incomplètes dans la configuration.")
    return None

def load_full_config():
    """
    Charge et fusionne les configurations simulation.yaml et events.yaml dans un seul dictionnaire.
    - simulation → sous clé 'simulation'
    - events → sous clé 'events'
    """
    simulation_cfg = load_simulation_config()
    simulation_only = simulation_cfg.get("simulation", {})
    if not simulation_only:
        print("[WARN] Aucune configuration 'simulation' trouvée dans simulation.yaml.")
    events_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'events.yaml')
    with open(events_path, 'r') as f:
        events_cfg = yaml.safe_load(f)
    general_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'global.yaml')
    with open(general_path, 'r') as f:
        general_cfg = yaml.safe_load(f)

    return {
        "simulation": simulation_only,
        "events": events_cfg,
        "general": general_cfg["general"]
    }

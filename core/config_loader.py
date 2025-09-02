import os
import logging
from pathlib import Path
import yaml
from core.decorators import deprecated

logger = logging.getLogger(__name__)

# Defaults (kept for backward compatibility; now resolved via _resolve_config_path)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
SIMULATION_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'simulation.yaml')

_cached_config = None
_cached_simulation_config = None

# ---------- Robust path resolver ----------
def _resolve_config_path(config_path: str) -> Path:
    """
    Resolve a config file path robustly when RS3 is called from another project.
    Priority:
      1) Absolute path
      2) RS3_CONFIG_DIR env var (join with config_path)
      3) Current Working Directory (CWD)
      4) RS3 repo root inferred from this file location (…/RoadSimulator3)
    """
    p = Path(config_path)
    if p.is_absolute() and p.exists():
        return p

    # 1) Env override
    env_dir = os.environ.get("RS3_CONFIG_DIR")
    if env_dir:
        p_env = Path(env_dir) / config_path
        if p_env.exists():
            return p_env

    # 2) CWD
    p_cwd = Path.cwd() / config_path
    if p_cwd.exists():
        return p_cwd

    # 3) Repo root from this file: core/ → parent is repo root
    repo_root = Path(__file__).resolve().parents[1]
    p_repo = repo_root / config_path
    if p_repo.exists():
        return p_repo

    # 4) Legacy: some callers pass only the filename; try repo_root/config/
    p_repo_cfg = repo_root / "config" / Path(config_path).name
    if p_repo_cfg.exists():
        return p_repo_cfg

    raise FileNotFoundError(f"Config file not found via CWD/ENV/RepoRoot resolution: {config_path}")

# ---------- Public API ----------
def load_config(config_path: str | None = None):
    """
    Charge le fichier de configuration principal.
    Résolution robuste du chemin (CWD, RS3_CONFIG_DIR, racine du repo).
    """
    global _cached_config
    if _cached_config is not None and config_path is None:
        return _cached_config

    if config_path is None:
        config_path = "config/config.yaml"

    resolved = _resolve_config_path(config_path)
    with resolved.open('r') as f:
        cfg = yaml.safe_load(f)

    if config_path == "config/config.yaml":
        _cached_config = cfg
    return cfg

def load_simulation_config(config_path: str | None = None):
    """
    Charge la configuration spécifique à la simulation (simulation.yaml).
    Résolution robuste du chemin (CWD, RS3_CONFIG_DIR, racine du repo).
    """
    global _cached_simulation_config
    if _cached_simulation_config is not None and config_path is None:
        return _cached_simulation_config

    if config_path is None:
        config_path = "config/simulation.yaml"

    resolved = _resolve_config_path(config_path)
    with resolved.open('r') as f:
        cfg = yaml.safe_load(f)

    if config_path == "config/simulation.yaml":
        _cached_simulation_config = cfg
    return cfg

# Fonctions spécifiques
@deprecated
def get_context_config():
    logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
    cfg = load_config()
    return cfg.get('context', {})

def load_full_config():
    """
    Charge et fusionne les configurations simulation.yaml, events.yaml et global.yaml dans un seul dictionnaire.
    - simulation → sous clé 'simulation'
    - events → sous clé 'events'
    - general → sous clé 'general'
    """
    simulation_cfg = load_simulation_config()
    simulation_only = simulation_cfg.get("simulation", {})
    if not simulation_only:
        logger.warning("Aucune configuration 'simulation' trouvée dans simulation.yaml.")

    events_path = _resolve_config_path('config/events.yaml')
    with events_path.open('r') as f:
        events_cfg = yaml.safe_load(f)

    general_path = _resolve_config_path('config/global.yaml')
    with general_path.open('r') as f:
        general_cfg = yaml.safe_load(f)

    return {
        "simulation": simulation_only,
        "events": events_cfg,
        "general": general_cfg.get("general", {}),
    }

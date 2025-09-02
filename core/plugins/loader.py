import os, importlib, logging, json, sys
from typing import List, Tuple, Dict, Any, Optional
from importlib.metadata import entry_points
from copy import deepcopy
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Robust path resolver (uses core.config_loader if available)
# ---------------------------------------------------------------------------
def _resolve_path(path: str | os.PathLike) -> Path:
    """
    Resolve a config file path robustly when RS3 is used as a library.
    Priority:
      1) Absolute path
      2) RS3_CONFIG_DIR env var (join with path)
      3) Current Working Directory (CWD)
      4) RS3 repo root inferred from this file location (…/RoadSimulator3)
    Returns a Path that may or may not exist (caller decides).
    """
    p = Path(path)
    if p.is_absolute():
        return p
    # try core.config_loader if present
    try:
        from core.config_loader import _resolve_config_path as _rcp  # type: ignore
        return _rcp(str(path))  # will raise if not found
    except Exception:
        pass

    # local fallback (non-raising)
    env_dir = os.environ.get("RS3_CONFIG_DIR")
    if env_dir:
        p_env = Path(env_dir) / path
        if p_env.exists():
            return p_env
    p_cwd = Path.cwd() / path
    if p_cwd.exists():
        return p_cwd
    repo_root = Path(__file__).resolve().parents[2]  # core/plugins/ -> repo root
    p_repo = repo_root / path
    if p_repo.exists():
        return p_repo
    # final best-effort
    return p_repo

# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

def _deep_merge(a: dict, b: dict) -> dict:
    out = deepcopy(a) if isinstance(a, dict) else {}
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out

def _merge_yaml_file(base: dict, path: str) -> dict:
    """
    Load YAML from `path` (resolved robustly) if it exists and deep-merge into base.
    - accepts relative paths (resolved via RS3_CONFIG_DIR/CWD/repo-root)
    - silently skips if yaml unavailable or file unreadable
    """
    if not path:
        return base
    try:
        resolved = _resolve_path(path)
        if not resolved.exists():
            return base
        if yaml is None:
            log.warning("[PLUGINS] PyYAML unavailable; cannot read %s", resolved)
            return base
        with resolved.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return _deep_merge(base, data)
    except Exception as e:
        log.warning("[PLUGINS] ignore config file %s (%s)", path, e)
        return base

# ---------------------------------------------------------------------------
# Plugin config merging
# ---------------------------------------------------------------------------
def load_plugin_configs(enrichers) -> dict:
    """
    Construit une mega-config résultant de:
      defaults.yaml (plugin)  <  overrides utilisateur (fichiers)  <  variables d'env
    Les chemins relatifs sont résolus via RS3_CONFIG_DIR / CWD / racine du repo RS3.
    """
    merged: Dict[str, Any] = {}

    # 1) defaults embedded in each plugin
    for enr in enrichers:
        try:
            defaults = enr.config_default()
            if isinstance(defaults, dict):
                merged = _deep_merge(merged, defaults)
        except Exception:
            pass

        # 2) user overrides provided *by the plugin* (paths can depend on env)
        try:
            paths = getattr(enr, "config_user_paths", lambda: [])()
            for path in paths or []:
                merged = _merge_yaml_file(merged, path)
        except Exception:
            pass

    # 3) global overrides via env-provided YAML files (colon-separated)
    env_files = os.environ.get("RS3_PLUGIN_CONFIGS", "").split(":")
    for path in filter(None, env_files):
        merged = _merge_yaml_file(merged, path)

    # 4) global overrides via inline JSON/YAML in env (optional)
    inline = os.environ.get("RS3_PLUGIN_CONFIG_JSON") or os.environ.get("RS3_PLUGIN_CONFIG_YAML")
    if inline:
        try:
            try:
                inline_data = json.loads(inline)
            except json.JSONDecodeError:
                inline_data = yaml.safe_load(inline) if yaml else {}
            if isinstance(inline_data, dict):
                merged = _deep_merge(merged, inline_data)
            else:
                log.warning("[PLUGINS] RS3_PLUGIN_CONFIG_* must be a mapping at the top level; got %s", type(inline_data))
        except Exception as e:
            log.warning("[PLUGINS] failed to parse RS3_PLUGIN_CONFIG_* (%s)", e)

    log.info("[PLUGINS] config merged (keys: %s)", ", ".join(sorted(merged.keys())))
    return merged

# ---------------------------------------------------------------------------
# Entry points discovery
# ---------------------------------------------------------------------------
def _iter_rs3_entry_points():
    """Yield entry points for group 'rs3.plugins' across importlib.metadata versions.
    - Python 3.10+/newer: EntryPoints.select(group=...)
    - Older importlib_metadata: mapping-like .get('group', [])
    - Fallback: filter by .group attribute
    """
    group = os.environ.get("RS3_PLUGIN_ENTRYPOINT_GROUP", "rs3.plugins")
    try:
        eps = entry_points()
        select = getattr(eps, "select", None)
        if callable(select):
            return list(eps.select(group=group))
        get = getattr(eps, "get", None)
        if callable(get):
            return list(get(group, []))
        return [ep for ep in eps if getattr(ep, "group", None) == group]
    except Exception:
        return []

# ---------------------------------------------------------------------------
# Dev-time plugin import helpers
# ---------------------------------------------------------------------------
def _import_from_fs(path: str):
    """
    Import a module from filesystem path (folder or .py file).
    Returns the module or raises.
    """
    import importlib.util
    p = Path(path)
    if p.is_dir():
        # treat as package directory
        spec = importlib.util.spec_from_file_location(p.name, p / "__init__.py")
    else:
        spec = importlib.util.spec_from_file_location(p.stem, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def _iter_dev_plugins() -> List[object]:
    """
    Iterate over modules declared in RS3_PLUGIN_PATHS.
    Accepts:
      - dotted module names (e.g. rs3_plugin_altitude)
      - filesystem paths to packages or .py files
    """
    out = []
    dev_paths = os.environ.get("RS3_PLUGIN_PATHS", "")
    for item in filter(None, dev_paths.split(":")):
        try:
            if "/" in item or item.endswith(".py"):
                mod = _import_from_fs(item)
            else:
                mod = importlib.import_module(item)
            out.append(mod)
        except Exception as e:
            log.warning("[PLUGINS] (dev) échec %s: %s", item, e)
    return out

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_plugins() -> Tuple[List[object], List[object]]:
    """
    Découvre les plugins via entry_points 'rs3.plugins' + chemins dev (RS3_PLUGIN_PATHS).
    Respecte les allow/deny lists via RS3_PLUGINS_ALLOWLIST / RS3_PLUGINS_DENYLIST (noms séparés par ',').
    """
    enrichers, runners = [], []

    allow = set(filter(None, os.environ.get("RS3_PLUGINS_ALLOWLIST", "").split(",")))
    deny  = set(filter(None, os.environ.get("RS3_PLUGINS_DENYLIST", "").split(",")))

    def _accept(name: str) -> bool:
        if allow and name not in allow:
            return False
        if name in deny:
            return False
        return True

    # 1) entry_points (pip installés)
    for ep in _iter_rs3_entry_points():
        try:
            plugin_cls = ep.load()
            plugin = plugin_cls()
            name = getattr(plugin, "name", ep.name)
            if not _accept(name):
                log.info("[PLUGINS] ignoré (policy): %s", name)
                continue
            if getattr(plugin, "kind", "") == "runner":
                runners.append(plugin)
            else:
                enrichers.append(plugin)
            log.info("[PLUGINS] chargé: %s (%s)", name, getattr(plugin, "license", "?"))
        except Exception as e:
            log.warning("[PLUGINS] échec chargement %s: %s", ep.name, e)

    # 2) chemins dev (séparés par ':')
    for mod in _iter_dev_plugins():
        try:
            plugin_cls = getattr(mod, "Plugin", None) or getattr(mod, "AltitudePlugin", None) or getattr(mod, "FleetPlugin", None)
            if not plugin_cls:
                log.warning("[PLUGINS] (dev) aucun Plugin/AltitudePlugin/FleetPlugin exporté dans %s", getattr(mod, "__name__", mod))
                continue
            plugin = plugin_cls()
            name = getattr(plugin, "name", getattr(mod, "__name__", "dev_plugin"))
            if not _accept(name):
                log.info("[PLUGINS] (dev) ignoré (policy): %s", name)
                continue
            (enrichers if getattr(plugin, "kind","")!="runner" else runners).append(plugin)
            log.info("[PLUGINS] (dev) chargé: %s", name)
        except Exception as e:
            log.warning("[PLUGINS] (dev) échec %s: %s", getattr(mod, "__name__", mod), e)

    return enrichers, runners

def merge_configs_into(base: Dict[str, Any], plugin_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict equal to deep-merge of base <- plugin_cfg (plugin wins)."""
    base = deepcopy(base) if isinstance(base, dict) else {}
    return _deep_merge(base, plugin_cfg or {})

__all__ = [
    "load_plugins",
    "load_plugin_configs",
    "merge_configs_into",
]
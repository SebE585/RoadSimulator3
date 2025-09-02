import os, importlib, logging, json
from typing import List, Tuple, Dict, Any
from importlib.metadata import entry_points

log = logging.getLogger(__name__)

import os, yaml
from copy import deepcopy

def _deep_merge(a: dict, b: dict) -> dict:
    out = deepcopy(a)
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out

def _merge_yaml_file(base: dict, path: str) -> dict:
    """Load YAML from `path` if it exists and deep-merge into base."""
    if not path or not os.path.exists(path):
        return base
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return _deep_merge(base, data)
    except Exception as e:
        log.warning("[PLUGINS] ignore config file %s (%s)", path, e)
        return base

def load_plugin_configs(enrichers) -> dict:
    """
    Construit une mega-config résultant de:
      defaults.yaml (plugin)  <  overrides utilisateur (fichiers)  <  variables d'env
    Sans commit dans le repo principal.
    """
    merged = {}

    # 1) defaults embedded in each plugin
    for enr in enrichers:
        try:
            defaults = enr.config_default()
            merged = _deep_merge(merged, defaults)
        except Exception:
            pass

        # 2) user overrides provided *by the plugin* (paths can depend on env)
        try:
            for path in getattr(enr, "config_user_paths", lambda: [])():
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
            # Try JSON first, then YAML
            try:
                inline_data = json.loads(inline)
            except json.JSONDecodeError:
                inline_data = yaml.safe_load(inline) or {}
            if isinstance(inline_data, dict):
                merged = _deep_merge(merged, inline_data)
            else:
                log.warning("[PLUGINS] RS3_PLUGIN_CONFIG_* must be a mapping at the top level; got %s", type(inline_data))
        except Exception as e:
            log.warning("[PLUGINS] failed to parse RS3_PLUGIN_CONFIG_* (%s)", e)

    log.info("[PLUGINS] config merged (keys: %s)", ", ".join(sorted(merged.keys())))
    return merged

def _iter_rs3_entry_points():
    """Yield entry points for group 'rs3.plugins' across importlib.metadata versions.
    - Python 3.10+/newer: EntryPoints.select(group=...)
    - Older importlib_metadata: mapping-like .get('group', [])
    - Fallback: filter by .group attribute
    """
    try:
        eps = entry_points()
        # Preferred API on modern Python
        select = getattr(eps, "select", None)
        if callable(select):
            return list(eps.select(group="rs3.plugins"))
        # Legacy mapping API
        get = getattr(eps, "get", None)
        if callable(get):
            return list(get("rs3.plugins", []))
        # Last resort: iterate and filter
        return [ep for ep in eps if getattr(ep, "group", None) == "rs3.plugins"]
    except Exception:
        return []

def load_plugins() -> Tuple[List[object], List[object]]:
    """Découvre les plugins via entry_points 'rs3.plugins' + chemins dev (RS3_PLUGIN_PATHS)."""
    enrichers, runners = [], []

    # 1) entry_points (pip installés) — compatible avec Python 3.8→3.12
    for ep in _iter_rs3_entry_points():
        try:
            plugin_cls = ep.load()
            plugin = plugin_cls()
            if getattr(plugin, "kind", "") == "enricher":
                enrichers.append(plugin)
            elif getattr(plugin, "kind", "") == "runner":
                runners.append(plugin)
            else:
                enrichers.append(plugin)  # par défaut
            log.info("[PLUGINS] chargé: %s (%s)", getattr(plugin, "name", ep.name), getattr(plugin, "license", "?"))
        except Exception as e:
            log.warning("[PLUGINS] échec chargement %s: %s", ep.name, e)

    # 2) chemins dev (séparés par ':')
    dev_paths = os.environ.get("RS3_PLUGIN_PATHS", "")
    for modname in filter(None, dev_paths.split(":")):
        try:
            mod = importlib.import_module(modname)
            plugin_cls = getattr(mod, "Plugin", None) or getattr(mod, "AltitudePlugin", None) or getattr(mod, "FleetPlugin", None)
            if plugin_cls:
                plugin = plugin_cls()
                (enrichers if getattr(plugin, "kind","")!="runner" else runners).append(plugin)
                log.info("[PLUGINS] (dev) chargé: %s", modname)
        except Exception as e:
            log.warning("[PLUGINS] (dev) échec %s: %s", modname, e)

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
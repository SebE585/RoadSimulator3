# core2/plugin_loader.py
from __future__ import annotations
import logging
from importlib.metadata import entry_points
from typing import List, Any, Optional

log = logging.getLogger(__name__)

def discover_external_stages(cfg: Optional[dict] = None) -> List[Any]:
    stages: List[Any] = []

    # 1) Entry points officiels
    try:
        eps = entry_points(group="rs3.plugins")
        found = [e.name for e in eps]
        log.info("[Plugins] EP trouvés (rs3.plugins): %s", found)

        for ep in eps:
            try:
                factory = ep.load()
                log.info("[Plugins] Chargement EP '%s' → %r", ep.name, factory)
                # factory peut être une fonction discover_stages(cfg) ou une classe Stage
                if callable(factory):
                    produced = factory(cfg)  # peut retourner une liste de stages ou un stage
                    if produced is None:
                        continue
                    if isinstance(produced, (list, tuple)):
                        stages.extend(list(produced))
                    else:
                        stages.append(produced)
                else:
                    log.warning("[Plugins] EP '%s' non-callable (%r), ignoré.", ep.name, factory)
            except Exception as e:
                log.warning("[Plugins] EP '%s' échec de load/appel: %s", ep.name, e)
    except Exception as e:
        log.warning("[Plugins] Découverte EP impossible: %s", e)

    # 2) Fallback import direct (utile en dev, si EP manquants)
    if not stages:
        try:
            import rs3_plugin_altitude_agpl.plugin as altmod
            if hasattr(altmod, "discover_stages"):
                produced = altmod.discover_stages(cfg)
                if produced:
                    if isinstance(produced, (list, tuple)):
                        stages.extend(list(produced))
                    else:
                        stages.append(produced)
                log.info("[Plugins] Fallback direct OK: altitude")
        except Exception as e:
            log.info("[Plugins] Fallback altitude KO: %s", e)

    # Log final
    if stages:
        names = [getattr(s, "name", s.__class__.__name__) for s in stages]
        log.info("[Plugins] Stages externes activés: %s", names)
    else:
        log.info("[Plugins] Aucun stage externe détecté.")

    return stages
from typing import Dict, List, Optional
import pandas as pd

class DatasetEnricherPlugin:
    """Un plugin qui enrichit un DataFrame et/ou ajoute des colonnes."""
    name: str = "base-enricher"

    def provides_schema_fragments(self) -> List[Dict]:
        """Retourne une liste d’objets YAML-like {'columns': [...]} à fusionner dans le schéma."""
        return []

    def apply(self, df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
        """Retourne un DF enrichi (peut ajouter des colonnes)."""
        return df


class RunnerPlugin:
    """Un plugin qui expose une commande CLI (ex: run_fleet)."""
    command: str = "noop"

    def run(self, argv: List[str], config: Optional[dict] = None) -> int:
        """Exécute la commande. Retourne un code de sortie."""
        return 0
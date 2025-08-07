from collections import defaultdict
import pandas as pd
from core.decorators import deprecated


class EventCounter:
    """
    Classe utilitaire pour suivre et résumer le nombre d'événements injectés
    dans une simulation à partir d'un DataFrame ou manuellement.

    Attributs :
        counts (defaultdict) : Dictionnaire des occurrences d'événements.
    """

    def __init__(self):
        """
        Initialise un compteur d'événements vide.
        """
        self.counts = defaultdict(int)

    def add(self, event_type: str, n: int = 1) -> None:
        """
        Ajoute un ou plusieurs événements d'un type donné.

        Args:
            event_type (str): Type d'événement (ex. 'freinage').
            n (int): Nombre à ajouter (défaut = 1).
        """
        self.counts[event_type] += n

    def get(self, event_type: str) -> int:
        """
        Retourne le nombre d'occurrences pour un type d'événement donné.

        Args:
            event_type (str): Nom de l'événement.

        Returns:
            int: Nombre d'occurrences.
        """
        return self.counts.get(event_type, 0)

    def total(self) -> int:
        """
        Calcule le total de tous les événements enregistrés.

        Returns:
            int: Somme totale des événements.
        """
        return sum(self.counts.values())

    @deprecated
    def reset(self) -> None:
        logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
        """
        Réinitialise complètement le compteur.
        """
        self.counts.clear()

    @deprecated
    def as_dict(self) -> dict:
        logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
        """
        Exporte les données de comptage sous forme de dictionnaire.

        Returns:
            dict: Copie du dictionnaire des comptes.
        """
        return dict(self.counts)

    def count_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Analyse un DataFrame et compte les occurrences de chaque événement.

        Args:
            df (pd.DataFrame): DataFrame contenant une colonne 'event'.
        """
        if "event" not in df.columns:
            return
        event_counts = df["event"].value_counts(dropna=True)
        for event_type, count in event_counts.items():
            if pd.notna(event_type) and str(event_type).strip():
                self.add(str(event_type), int(count))

    def summary(self) -> str:
        """
        Produit un résumé textuel des événements comptés.

        Returns:
            str: Chaîne contenant un récapitulatif lisible.
        """
        if not self.counts:
            return "\n[AUCUN ÉVÉNEMENT INJECTÉ]"
        lines = ["\n[RÉCAPITULATIF ÉVÉNEMENTS INJECTÉS]", ""]
        for k, v in sorted(self.counts.items()):
            lines.append(f"- {k:<15} : {v:>5}")
        return "\n".join(lines)

    def show(self, label: str = None) -> None:
        """
        Affiche le résumé des événements injectés avec un label optionnel.

        Args:
            label (str, optional): Titre à afficher avant le résumé.
        """
        if label:
            print(f"\n[{label}]")
        print(self.summary())
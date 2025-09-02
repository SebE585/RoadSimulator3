

import logging
import pandas as pd

logger = logging.getLogger(__name__)

def apply_delivery_markers(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """
    Ajoute les colonnes in_delivery (bool 0/1) et delivery_state (enum) à partir des événements de début/fin livraison.

    Règles:
      - Si l'événement contient 'start_delivery', on bascule en mode livraison (in_delivery=1, delivery_state='on_delivery')
      - Si l'événement contient 'end_delivery', on sort de la livraison (in_delivery=0, delivery_state='in_vehicle')
      - Sinon, la valeur est propagée depuis l'état précédent.

    Si aucune info disponible, les colonnes sont créées avec valeurs par défaut:
      - in_delivery = 0
      - delivery_state = "in_vehicle"
    """
    if df is None or df.empty:
        df["in_delivery"] = []
        df["delivery_state"] = []
        return df

    in_delivery = []
    delivery_state = []
    current_state = 0
    current_label = "in_vehicle"

    for ev in df.get("event", pd.Series([None] * len(df))):
        if isinstance(ev, str) and "start_delivery" in ev:
            current_state = 1
            current_label = "on_delivery"
        elif isinstance(ev, str) and "end_delivery" in ev:
            current_state = 0
            current_label = "in_vehicle"
        in_delivery.append(current_state)
        delivery_state.append(current_label)

    df["in_delivery"] = pd.Series(in_delivery, index=df.index, dtype="int8")
    df["delivery_state"] = pd.Series(delivery_state, index=df.index, dtype="object")

    logger.debug("Delivery markers applied (start/end events) → in_delivery/delivery_state")
    return df
import pandas as pd

def inject_opening_for_deliveries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject 'ouverture' (door opening) events just after each delivery stop (stop_xxx or wait_xxx).

    Assumptions:
    - Deliveries are labeled 'stop_xxx' or 'wait_xxx' in the 'event' column.
    - An opening occurs 1 second (10 rows at 10Hz) after each such delivery event.
    - Does not inject openings for generic 'stop' or 'wait' not related to deliveries.

    Returns:
    - DataFrame with 'event' column updated to include 'ouverture' events.
    """
    if "event" not in df.columns:
        return df
    df = df.copy()
    delivery_mask = df["event"].astype(str).str.match(r"(stop|wait)_[0-9]+")
    delivery_indices = df[delivery_mask].index

    for idx in delivery_indices:
        ouverture_idx = idx + 10  # 1s later at 10Hz
        if ouverture_idx < len(df):
            current_event = df.at[ouverture_idx, "event"]
            if pd.isna(current_event):
                df.at[ouverture_idx, "event"] = "ouverture"
            elif "ouverture" not in str(current_event):
                df.at[ouverture_idx, "event"] = f"{current_event}|ouverture"

    return df
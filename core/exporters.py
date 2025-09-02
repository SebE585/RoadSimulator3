# core/exporters.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import yaml
import pandas as pd
from typing import Dict, List, Optional


def _load_schema_columns(schema_path: str) -> List[str]:
    """Return ordered column names from dataset_schema.yaml."""
    with open(schema_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cols = data.get("columns", [])
    return [c.get("name") for c in cols if isinstance(c, dict) and c.get("name")]


def enforce_schema_order(
    df: pd.DataFrame,
    cfg: Optional[Dict] = None,
    *,
    schema_path: Optional[str] = None,
    drop_extras: Optional[bool] = None,
    ensure_altitude_alias: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Reorder DataFrame columns according to dataset_schema.yaml, add missing ones,
    ensure presence of `altitude_m`, and (optionally) drop extra columns.

    Parameters
    ----------
    df : DataFrame
        Input frame.
    cfg : dict, optional (backward-compat)
        May contain keys: "schema_path", "drop_extras", "ensure_altitude_alias".
    schema_path : str, optional
        Path to YAML schema. Overrides cfg["schema_path"] if provided.
    drop_extras : bool, optional
        If True, keep only schema columns. Defaults to True (v1.0 behavior).
    ensure_altitude_alias : bool, optional
        If True (default), create `altitude_m` from `altitude` if missing.

    Returns
    -------
    DataFrame
        A new frame with enforced order and schema guarantees.
    """
    if df is None or df.empty:
        return df

    cfg = cfg or {}
    schema_path = schema_path or cfg.get("schema_path", "config/dataset_schema.yaml")
    drop_extras = drop_extras if drop_extras is not None else cfg.get("drop_extras", True)
    ensure_altitude_alias = (
        ensure_altitude_alias if ensure_altitude_alias is not None else cfg.get("ensure_altitude_alias", True)
    )

    ordered = _load_schema_columns(schema_path)
    if not ordered:
        # No schema → return as-is
        return df

    out = df.copy()

    # Ensure altitude_m presence: alias from 'altitude' if needed
    if ensure_altitude_alias and ("altitude_m" not in out.columns):
        if "altitude" in out.columns:
            out["altitude_m"] = out["altitude"]
        else:
            out["altitude_m"] = pd.NA

    # Add any missing schema columns as NA
    for col in ordered:
        if col not in out.columns:
            out[col] = pd.NA

    if drop_extras:
        # Keep exactly the schema columns, in order
        out = out[ordered]
    else:
        # Keep extras appended after schema columns
        extras = [c for c in out.columns if c not in ordered]
        out = out[ordered + extras]

    return out


__all__ = ["enforce_schema_order"]
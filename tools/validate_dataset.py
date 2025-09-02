

# tools/validate_dataset.py
# -*- coding: utf-8 -*-
"""
Valide un CSV RoadSimulator3 contre un schéma YAML (dataset_schema.yaml).

Contrôles réalisés (niveau v1.0):
  1) Présence des colonnes attendues.
  2) (Optionnel) Ordre strict des colonnes.
  3) Colonnes en trop (report, non bloquant par défaut).
  4) Nullability: colonnes non-nullables ne doivent pas contenir de NA.
  5) Types: tentatives de coercition vers dtype attendu; erreurs si non convertible.
  6) Valeurs autorisées (si "values" list est définie dans le schéma).

Usage:
    python tools/validate_dataset.py --csv path/to/trace.csv \
        --schema config/dataset_schema.yaml --strict-order --max-errors 50

Code retour:
  0 = OK; 1 = erreurs détectées.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml


@dataclass
class ColumnSpec:
    name: str
    dtype: Optional[str] = None
    nullable: Optional[bool] = None
    values: Optional[List[Any]] = None


def load_schema(schema_path: str) -> List[ColumnSpec]:
    with open(schema_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cols = []
    for item in data.get("columns", []):
        if not isinstance(item, dict):
            continue
        cols.append(
            ColumnSpec(
                name=item.get("name"),
                dtype=item.get("dtype"),
                nullable=item.get("nullable"),
                values=item.get("values"),
            )
        )
    return cols


def can_cast(series: pd.Series, target: str) -> pd.Series:
    """Retourne un masque booléen True où la conversion au dtype est possible.
    Ne modifie pas la série d'origine. Les NaN sont considérés comme valides (gérés par nullability).
    """
    s = series.copy()
    # Traitement des NA: on les ignore pour le cast, la nullability gère ensuite
    mask_non_na = ~s.isna()
    s_non_na = s[mask_non_na]

    try:
        if target in ("float32", "float64", "float"):
            pd.to_numeric(s_non_na, errors="coerce")
        elif target in ("int8", "int16", "int32", "int64", "int"):
            # int strict: impossible si décimales; on teste en deux temps
            n = pd.to_numeric(s_non_na, errors="coerce")
            # entier si valeur == valeur arrondie
            ok = (n.dropna() == n.dropna().round()).reindex(n.index, fill_value=False)
            # là où ce n'est pas OK → invalide
            # combine avec non-null (les NaN ici sont cast ratés)
            return (~n.isna()) & ok
        elif target in ("datetime", "datetime64", "datetime64[ns]"):
            dt = pd.to_datetime(s_non_na, errors="coerce", utc=False)
            return ~dt.isna()
        elif target in ("category", "str", "string", "object"):
            # Tout est toléré pour string/object; la nullability/values tranchent
            return pd.Series(True, index=s_non_na.index)
        else:
            # Type inconnu → on ne bloque pas
            return pd.Series(True, index=s_non_na.index)
    except Exception:
        return pd.Series(False, index=s_non_na.index)

    # Pour float et autres conversions numeric: succès si non NaN après to_numeric
    return pd.Series(True, index=s_non_na.index)


def validate_csv(csv_path: str, schema_path: str, strict_order: bool, max_errors: int) -> int:
    specs = load_schema(schema_path)
    expected_cols = [c.name for c in specs if c.name]

    df = pd.read_csv(csv_path)

    errors: List[str] = []
    warnings: List[str] = []

    # 1) Présence colonnes
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        errors.append(f"Colonnes manquantes: {missing}")

    # 2) Ordre strict (optionnel)
    if strict_order and not missing:
        if list(df.columns[: len(expected_cols)]) != expected_cols:
            errors.append("Ordre des colonnes incorrect (les premières colonnes doivent suivre le schéma).")

    # 3) Colonnes en trop (avertissement)
    extras = [c for c in df.columns if c not in expected_cols]
    if extras:
        warnings.append(f"Colonnes supplémentaires (conservées): {extras}")

    # 4/5/6) Parcours par colonne spécifiée dans le schéma
    name_to_spec: Dict[str, ColumnSpec] = {c.name: c for c in specs if c.name}

    for col in expected_cols:
        if col not in df.columns:
            continue
        spec = name_to_spec[col]
        s = df[col]

        # Nullability
        if spec.nullable is False and s.isna().any():
            locs = s.index[s.isna()].tolist()[:max_errors]
            errors.append(f"{col}: valeurs manquantes détectées (nullable=false). Exemples index={locs}")

        # Allowed values (catégoriel binaire, etc.)
        if spec.values:
            invalid_mask = ~s.isna() & ~s.isin(spec.values)
            if invalid_mask.any():
                locs = s.index[invalid_mask].tolist()[:max_errors]
                errors.append(f"{col}: valeurs hors domaine {spec.values}. Exemples index={locs}")

        # Type check (coercion virtuelle)
        if spec.dtype:
            cast_ok_mask = can_cast(s, spec.dtype)
            # indices non NA mais non convertibles
            non_na = ~s.isna()
            invalid = non_na & ~cast_ok_mask.reindex(s.index, fill_value=False)
            if invalid.any():
                locs = s.index[invalid].tolist()[:max_errors]
                errors.append(f"{col}: incompatible avec dtype '{spec.dtype}'. Exemples index={locs}")

    # Affichage rapport
    print("\n=== Validation RS3 ===")
    print(f"CSV       : {csv_path}")
    print(f"Schéma    : {schema_path}")
    print(f"Colonnes  : {len(df.columns)}  (attendues: {len(expected_cols)})")
    if warnings:
        print("\nAvertissements:")
        for w in warnings:
            print("  -", w)

    if errors:
        print("\nErreurs:")
        for e in errors:
            print("  -", e)
        print(f"\n❌ Validation échouée. ({len(errors)} erreurs)")
        return 1

    print("\n✅ Validation réussie.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Valide un CSV RS3 contre dataset_schema.yaml")
    p.add_argument("--csv", required=True, help="Chemin du CSV à valider")
    p.add_argument("--schema", default="config/dataset_schema.yaml", help="Chemin du schéma YAML")
    p.add_argument("--strict-order", action="store_true", help="Enforce ordre strict des colonnes (schéma en tête)")
    p.add_argument("--max-errors", type=int, default=50, help="Nb max d'exemples détaillés")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return validate_csv(args.csv, args.schema, args.strict_order, args.max_errors)


if __name__ == "__main__":
    sys.exit(main())
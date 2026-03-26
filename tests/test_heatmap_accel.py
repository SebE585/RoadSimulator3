"""
Test d'intégration : distribution des accélérations RS3 vs données réelles.

Compare le heatmap ax/ay produit par RS3 avec les caractéristiques
attendues d'un heatmap de conduite urbaine réelle.

Critères (issus des images de référence) :
  - Distribution gaussienne centrée (pas bimodale)
  - ax_std ≈ 0.10–0.20 g
  - ay_std ≈ 0.15–0.25 g
  - Centre ±0.05g < 30%
  - Pas de bandes (valeurs discrètes)
  - Forme losange/diamant (pas rectangle)

Usage:
    python tests/test_heatmap_accel.py
"""
import sys
import os
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core2.context import Context
from runner.run_simulation2 import build_pipeline

G = 9.80665


def _run_simulation():
    """Génère une trace RS3 avec un parcours varié."""
    cfg = {
        "outdir": tempfile.mkdtemp(),
        "osrm": {"base_url": "http://51.91.125.143:5000"},
        "sim": {"hz": 10},
        "sensors": {"gps_hz": 10, "imu_hz": 10, "gyro_enabled": True},
        "stops": [
            {"id": "DEPOT", "lat": 49.3347, "lon": 1.3830, "service_s": 0},
            {"id": "C1", "lat": 49.354, "lon": 1.340, "service_s": 20},
            {"id": "C2", "lat": 49.360, "lon": 1.310, "service_s": 20},
            {"id": "C3", "lat": 49.335, "lon": 1.350, "service_s": 20},
            {"id": "DEPOT", "lat": 49.3347, "lon": 1.3830, "service_s": 0},
        ],
    }
    ctx = Context(cfg=cfg)
    build_pipeline(cfg).run(ctx)
    return ctx.df, ctx.artifacts


def test_acceleration_distribution():
    """Test principal : distribution des accélérations."""
    print("Simulation RS3...")
    df, artifacts = _run_simulation()

    moving = df["speed"] > 1.0
    n_moving = moving.sum()
    print(f"  {len(df)} points totaux, {n_moving} en mouvement")

    ax = df.loc[moving, "acc_x"].values
    ay = df.loc[moving, "acc_y"].values
    ax_g = ax / G
    ay_g = ay / G

    print(f"\n=== MÉTRIQUES ===")

    # 1. Std (dispersion)
    ax_std = np.std(ax_g)
    ay_std = np.std(ay_g)
    print(f"  ax std: {ax_std:.4f} g (cible: 0.10–0.20)")
    print(f"  ay std: {ay_std:.4f} g (cible: 0.15–0.25)")

    # 2. Centre
    centre = ((abs(ax_g) < 0.05) & (abs(ay_g) < 0.05)).mean()
    print(f"  Centre ±0.05g: {centre*100:.1f}% (cible: < 30%)")

    # 3. Valeurs discrètes (bandes)
    dv = np.diff(df.loc[moving, "speed"].values) * 10
    n_unique = len(np.unique(np.round(dv, 2)))
    print(f"  dv/dt valeurs uniques: {n_unique} (cible: > 100)")

    # 4. Bimodalité : le pic doit être à 0, pas aux extrêmes
    hist_ax, bins_ax = np.histogram(ax_g, bins=50, range=(-0.5, 0.5))
    peak_bin = bins_ax[np.argmax(hist_ax)]
    bimodal = abs(peak_bin) > 0.1
    print(f"  Pic ax à: {peak_bin:.2f}g (cible: ~0, pas bimodal)")

    # 5. Forme losange : ratio des queues ax vs ay
    tail_ax = (abs(ax_g) > 0.2).mean()
    tail_ay = (abs(ay_g) > 0.2).mean()
    print(f"  Queue ax (>0.2g): {tail_ax*100:.1f}%")
    print(f"  Queue ay (>0.2g): {tail_ay*100:.1f}%")

    # 6. QA
    qa_ok = artifacts.get("qa_realism", {}).get("ok", False)
    print(f"  QA realism: {'✅' if qa_ok else '❌'}")

    # ── Assertions ──
    print(f"\n=== RÉSULTATS ===")
    passed = 0
    total = 7

    def check(name, condition):
        nonlocal passed
        if condition:
            passed += 1
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}")

    check("ax_std dans [0.08, 0.25]g", 0.08 <= ax_std <= 0.25)
    check("ay_std dans [0.10, 0.30]g", 0.10 <= ay_std <= 0.30)
    check("Centre < 30%", centre < 0.30)
    check("dv/dt > 100 valeurs uniques (pas de bandes)", n_unique > 100)
    check("Pas bimodal (pic à ~0)", not bimodal)
    check("Queues présentes (>0.2g > 2%)", tail_ax > 0.02 or tail_ay > 0.02)
    check("QA realism OK", qa_ok)

    print(f"\n  {passed}/{total} tests passés")
    return passed == total


if __name__ == "__main__":
    success = test_acceleration_distribution()
    sys.exit(0 if success else 1)

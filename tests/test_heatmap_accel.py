"""
Test d'intégration : distribution des accélérations RS3 vs données réelles.

Compare les critères de sévérité de conduite produits par RS3
avec les caractéristiques attendues d'un profil MOYEN :
  - Std spatiale (pondérée distance V·dt) :
      ax_std ≈ 0.069g, ay_std ≈ 0.094g
  - Distribution gaussienne centrée (pas bimodale)
  - Pas de bandes (valeurs discrètes)
  - Forme losange/diamant

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
from core2.accel_stats import compute_severity_criteria
from runner.simulate import build_pipeline

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
    """Test principal : distribution des accélérations (critères de sévérité)."""
    print("Simulation RS3...")
    df, artifacts = _run_simulation()

    # Critères de sévérité (pondérés distance)
    criteria = compute_severity_criteria(df, hz=10.0)

    print(f"  {len(df)} points totaux, {criteria['n_valid_points']} valides (en mouvement + physiques)")
    print(f"  Distance totale: {criteria['total_distance_m']:.0f} m")
    print(f"  Points filtrés (non-physiques): {criteria['n_filtered_non_physical']}")

    print(f"\n=== CRITÈRES DE SÉVÉRITÉ (pondérés distance) ===")
    ax_std = criteria["std_gx_spatial"]
    ay_std = criteria["std_gy_spatial"]
    print(f"  Std Gx (spatial): {ax_std:.4f} g (réf MOYEN: 0.069)")
    print(f"  Std Gy (spatial): {ay_std:.4f} g (réf MOYEN: 0.094)")
    print(f"  Mean Gx (spatial): {criteria['mean_gx_spatial']:.4f} g")
    print(f"  Mean Gy (spatial): {criteria['mean_gy_spatial']:.4f} g")
    print(f"  P0.1% Gx: [{criteria['p01_gx_low']:.4f}, {criteria['p01_gx_high']:.4f}] g")
    print(f"  P0.1% Gy: [{criteria['p01_gy_low']:.4f}, {criteria['p01_gy_high']:.4f}] g")

    print(f"\n=== COMPARAISON temporel vs spatial ===")
    print(f"  Std Gx temporel: {criteria['std_gx_temporal']:.4f} g")
    print(f"  Std Gy temporel: {criteria['std_gy_temporal']:.4f} g")

    # Métriques complémentaires (forme, bandes, bimodalité)
    moving = df["speed"].values > 0.5
    ax_g = df.loc[moving, "acc_x"].values / G
    ay_g = df.loc[moving, "acc_y"].values / G

    centre = ((abs(ax_g) < 0.05) & (abs(ay_g) < 0.05)).mean()
    print(f"\n  Centre +/-0.05g: {centre*100:.1f}%")

    dv = np.diff(df.loc[moving, "speed"].values) * 10
    n_unique = len(np.unique(np.round(dv, 2)))
    print(f"  dv/dt valeurs uniques: {n_unique}")

    hist_ax, bins_ax = np.histogram(ax_g, bins=50, range=(-0.5, 0.5))
    peak_bin = bins_ax[np.argmax(hist_ax)]
    bimodal = abs(peak_bin) > 0.1
    print(f"  Pic ax à: {peak_bin:.2f}g")

    tail_ax = (abs(ax_g) > 0.1).mean()
    tail_ay = (abs(ay_g) > 0.1).mean()
    print(f"  Queue ax (>0.1g): {tail_ax*100:.1f}%")
    print(f"  Queue ay (>0.1g): {tail_ay*100:.1f}%")

    qa_ok = artifacts.get("qa_realism", {}).get("ok", False)
    print(f"  QA realism: {'OK' if qa_ok else 'KO'}")

    # ── Assertions ──
    print(f"\n=== RÉSULTATS ===")
    passed = 0
    total = 7

    def check(name, condition):
        nonlocal passed
        if condition:
            passed += 1
            print(f"  OK  {name}")
        else:
            print(f"  KO  {name}")

    check("Std Gx spatial dans [0.04, 0.10]g", 0.04 <= ax_std <= 0.10)
    check("Std Gy spatial dans [0.04, 0.15]g", 0.04 <= ay_std <= 0.15)
    check("Centre < 30%", centre < 0.30)
    check("dv/dt > 100 valeurs uniques (pas de bandes)", n_unique > 100)
    check("Pas bimodal (pic à ~0)", not bimodal)
    check("Queues présentes (>0.1g > 1%)", tail_ax > 0.01 or tail_ay > 0.01)
    check("QA realism OK", qa_ok)

    print(f"\n  {passed}/{total} tests passés")
    return passed == total


if __name__ == "__main__":
    success = test_acceleration_distribution()
    sys.exit(0 if success else 1)

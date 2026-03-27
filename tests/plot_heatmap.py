"""
Génère les bi-histogrammes ax/ay pour les 3 profils conducteur.

Conforme aux méthodes standard d'analyse de sévérité de conduite :
  - Pondération par la distance V(t)·dt
  - Taille de classe 0.02g, plage [-1, 1]g
  - Normalisation par la distance totale
  - Std spatiale (pondérée distance)

Usage: python tests/plot_heatmap.py
"""
import sys, os, tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core2.context import Context
from core2.accel_stats import (
    compute_severity_criteria,
    build_bihistogram,
    _incremental_distance,
    filter_non_physical,
)
from runner.simulate import build_pipeline

G = 9.80665

STOPS = [
    {"id": "DEPOT", "lat": 49.3347, "lon": 1.3830, "service_s": 0},
    {"id": "C1", "lat": 49.354, "lon": 1.340, "service_s": 20},
    {"id": "C2", "lat": 49.360, "lon": 1.310, "service_s": 20},
    {"id": "C3", "lat": 49.335, "lon": 1.350, "service_s": 20},
    {"id": "DEPOT", "lat": 49.3347, "lon": 1.3830, "service_s": 0},
]

REF = {
    "severe": {"gx": 0.069, "gy": 0.130},
    "moyen":  {"gx": 0.069, "gy": 0.094},
    "doux":   {"gx": 0.056, "gy": 0.053},
}

TITLES = {"severe": "SÉVÈRE", "moyen": "MOYEN", "doux": "DOUX"}


def run_profile(profile: str):
    cfg = {
        "outdir": tempfile.mkdtemp(),
        "osrm": {"base_url": "http://51.91.125.143:5000"},
        "sim": {"hz": 10},
        "sensors": {"gps_hz": 10, "imu_hz": 10, "gyro_enabled": True},
        "stops": STOPS,
        "driving_dynamics": {"profile": profile},
    }
    ctx = Context(cfg=cfg)
    build_pipeline(cfg).run(ctx)
    return ctx.df


def main():
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for i, profile in enumerate(["severe", "moyen", "doux"]):
        print(f"\n{'='*40}")
        print(f"  Profil: {TITLES[profile]}")
        print(f"{'='*40}")
        df = run_profile(profile)

        # Critères de sévérité (pondérés distance)
        criteria = compute_severity_criteria(df, hz=10.0)
        ref = REF[profile]

        print(f"  Std Gx (spatial) = {criteria['std_gx_spatial']:.5f} g  (réf: {ref['gx']:.3f})")
        print(f"  Std Gy (spatial) = {criteria['std_gy_spatial']:.5f} g  (réf: {ref['gy']:.3f})")
        print(f"  Std Gx (temporel) = {criteria['std_gx_temporal']:.5f} g")
        print(f"  Std Gy (temporel) = {criteria['std_gy_temporal']:.5f} g")
        print(f"  Distance totale  = {criteria['total_distance_m']:.0f} m")
        print(f"  Points filtrés (non-physiques) = {criteria['n_filtered_non_physical']}")
        print(f"  P0.1% Gx: [{criteria['p01_gx_low']:.4f}, {criteria['p01_gx_high']:.4f}] g")
        print(f"  P0.1% Gy: [{criteria['p01_gy_low']:.4f}, {criteria['p01_gy_high']:.4f}] g")

        # Bi-histogramme pondéré distance
        speed = df["speed"].to_numpy(dtype=float)
        gx = df["acc_x"].to_numpy(dtype=float) / G
        gy = df["acc_y"].to_numpy(dtype=float) / G
        dist = _incremental_distance(speed, dt=0.1)
        moving = speed > 0.5
        phys = filter_non_physical(gx, gy)
        valid = moving & phys & np.isfinite(gx) & np.isfinite(gy)

        hist, xedges, yedges = build_bihistogram(
            gx[valid], gy[valid], dist[valid],
            range_g=0.5, class_size_g=0.02,
        )

        ax = axes[i]
        # log10(freq) comme sur l'image de référence
        hist_log = hist.copy()
        hist_log[hist_log <= 0] = np.nan

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(
            hist_log.T, extent=extent, origin="lower", aspect="equal",
            cmap="jet",
            norm=LogNorm(vmin=np.nanmin(hist_log[hist_log > 0]) if np.any(hist_log > 0) else 1e-8,
                         vmax=np.nanmax(hist_log) if np.any(np.isfinite(hist_log)) else 1),
        )
        ax.set_xlabel("gamma Y — Lat (g)", fontsize=11)
        if i == 0:
            ax.set_ylabel("gamma X — Long (g)", fontsize=11)
        title_color = "red" if profile == "severe" else ("orange" if profile == "moyen" else "green")
        ax.set_title(
            f"{TITLES[profile]}\n"
            f"Std Gx = {criteria['std_gx_spatial']:.5f} g    Std Gy = {criteria['std_gy_spatial']:.5f} g",
            fontsize=12, fontweight="bold", color=title_color,
        )
        ax.axhline(0, color="white", lw=0.5, alpha=0.5)
        ax.axvline(0, color="white", lw=0.5, alpha=0.5)
        plt.colorbar(im, ax=ax, shrink=0.8, label="log10(freq)")

    plt.suptitle(
        "RS3 — Profils conducteur (bi-histogramme pondéré distance)",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    out_path = os.path.join(tempfile.mkdtemp(), "heatmap_3profils.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nHeatmap 3 profils -> {out_path}")
    os.system(f'open "{out_path}"')


if __name__ == "__main__":
    main()

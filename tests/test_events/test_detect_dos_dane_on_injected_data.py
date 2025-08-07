import sys
import os
REINJECT_ALWAYS = True  # 🔁 Forçage de la réinjection pour debug
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import matplotlib.pyplot as plt
from simulator.detectors import detect_dos_dane
from simulator.events.generation import generate_dos_dane
from simulator.events.generation import get_available_indices
from core.config_loader import load_full_config
from pathlib import Path

def test_detect_dos_dane_on_injected_data():
    """
    Teste la détection des dos d’âne sur des données simulées avec événements injectés.
    """
    # Charger les données simulées
    trace_path = Path(os.environ.get("TRACE_CSV_PATH", "out/simulation_20250725_083424/trace.csv"))
    print(f"[TEST] Chargement du fichier de simulation : {trace_path}")
    df = pd.read_csv(trace_path)
    print(f"[TEST] Données chargées : {df.shape}")

    # Réinjection explicite des dos d’âne réalistes si gyro_z est nul
    if REINJECT_ALWAYS or "gyro_z" not in df.columns or df["gyro_z"].abs().max() < 0.01:
        print("[TEST] gyro_z inexistant ou nul → Réinjection des dos d’âne...")
        df["gyro_z"] = 0.0  # créer la colonne si absente
        injection_indices = get_available_indices(df, spacing=2000, count=5)
        print(f"[TEST] ✅ Réinjection forcée faite aux indices : {injection_indices}")
        config = load_full_config()
        df = generate_dos_dane(df, injection_indices=injection_indices, config=config)
        df.to_csv(trace_path, index=False)
        print("[TEST] Données sauvegardées après réinjection.")

    # Appliquer la détection avec des seuils plus permissifs pour valider l’efficacité du détecteur
    detected_flag, detected_indices = detect_dos_dane(df, acc_z_thresh=0.8, gyro_thresh=0.3, window_pts=5, refractory=15)
    if not detected_flag:
        detected_indices = []

    # Regrouper les indices injectés en événements distincts
    def group_injected_indices(indices, min_gap=20):
        if not indices:
            return []
        indices = sorted(indices)
        groups = [[indices[0]]]
        for idx in indices[1:]:
            if idx - groups[-1][-1] > min_gap:
                groups.append([idx])
            else:
                groups[-1].append(idx)
        return [g[len(g)//2] for g in groups]  # milieu de chaque groupe

    raw_injected = df[df["event"] == "dos_dane"].index.tolist()
    print(f"[TEST] Indices bruts injectés (dos_dane) : {raw_injected[:10]}... total: {len(raw_injected)}")
    grouped_injected = group_injected_indices(raw_injected)
    print(f"[TEST] Groupes injectés centrés (dos_dane) : {grouped_injected}")
    injected = set(grouped_injected)

    # Tracer toutes les signatures injectées pour vérification manuelle
    for i, idx in enumerate(sorted(injected)):
        window = df.iloc[max(0, idx - 15): idx + 15]
        acc = window["acc_z"].values
        gyro = window["gyro_z"].values
        print(f"[DEBUG] Signature {i+1} – Index {idx}")
        print(f"         acc_z: min={acc.min():.2f}, max={acc.max():.2f}, Δ={acc.max()-acc.min():.2f}")
        print(f"         gyro_z: min={gyro.min():.2f}, max={gyro.max():.2f}, Δ={gyro.max()-gyro.min():.2f}, mean={gyro.mean():.2f}")
        plt.figure(figsize=(6, 3))
        plt.plot(acc, label="acc_z", color='tab:blue')
        plt.plot(gyro, label="gyro_z", color='tab:orange')
        plt.title(f"Signature injectée {i+1} – Index {idx}")
        plt.axvline(15, color='gray', linestyle='--', alpha=0.5)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Tracer un exemple injecté pour diagnostic
    if injected:
        idx = next(iter(injected))
        window = df.iloc[max(0, idx - 15): idx + 15]
        window[["acc_z", "gyro_z"]].plot(title=f"Signature injectée autour de l’index {idx}")
        plt.grid(True)
        plt.show()

    from itertools import chain
    detected = set(chain.from_iterable([i] if isinstance(i, int) else i for i in detected_indices))
    print(f"[TEST] Indices détectés (bruts) : {sorted(detected_indices)}")
    print(f"[TEST] Indices détectés (flat) : {sorted(detected)}")

    # Intersection (tolérance à +/- 5)
    detected_matches = sum(any(abs(i - d) <= 5 for d in detected) for i in injected)

    print(f"✅ Dos d’âne injectés : {len(injected)}")
    print(f"✅ Dos d’âne détectés : {len(detected)}")
    print(f"🎯 Détections correctes (±5 points) : {detected_matches}")

    assert detected_matches >= min(3, len(injected)), "La détection n’a pas trouvé assez de dos d’âne."

    # Optionnel : afficher un exemple détecté
    if detected:
        idx = next(iter(detected))
        window = df.iloc[max(0, idx - 15): idx + 15]
        plt.plot(window["acc_z"].values, label="acc_z")
        plt.plot(window["gyro_z"].values, label="gyro_z")
        plt.legend()
        plt.title(f"Signature dos d’âne autour de l’index {idx}")
        plt.grid(True)
        plt.show()

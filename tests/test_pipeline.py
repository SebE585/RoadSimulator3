import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator.events import inject_events_on_route
from simulator.detectors import detect_all_events, detect_initial_acceleration, detect_final_deceleration
from check.check_realism import check_realism
from core.osrm_utils import simulate_route_via_osrm
from core.postprocessing import finalize_trajectory

HZ = 10
STEP_M = 0.83

def normalize_detection_result(x):
    if isinstance(x, (bool, np.bool_)):
        return x
    # liste, np.array, pd.Series -> True si au moins un True
    try:
        import numpy as np
        import pandas as pd
        if isinstance(x, (list, np.ndarray, pd.Series)):
            return bool(np.any(x))
    except ImportError:
        pass
    return bool(x)

def test_pipeline_consistency():
    # 1. Génération d'une trajectoire simple
    df = simulate_route_via_osrm(
        cities_coords=[
            (49.2738, 1.2127),
            (49.4431, 1.0993),
            (49.3568, 1.2342),
            (49.3653, 1.2361),
            (49.3305, 1.1811),
            (49.3364, 1.1733),
        ],
        hz=HZ,
        step_m=STEP_M
    )

    # 2. Injection des événements
    df = inject_events_on_route(df, hz=HZ)

    # 3. Détection rapide des phases critiques avant post-traitement
    initial_detected = detect_initial_acceleration(df)
    final_detected = detect_final_deceleration(df)
    print(f"[DEBUG] Accélération initiale détectée juste après injection : {'✅' if initial_detected else '❌'}")
    print(f"[DEBUG] Décélération finale détectée juste après injection : {'✅' if final_detected else '❌'}")

    # 4. Finalisation
    df = finalize_trajectory(df, hz=HZ)

    # 5. Analyse par signatures
    detected_signatures = detect_all_events(df)

    # 6. Analyse par labels (event)
    realism, _ = check_realism(df)

    # 7. Comparaison cohérente avec normalisation
    mapping = {
        'acceleration': '⬆️ Accélération détectée',
        'freinage': '🛑 Freinage détecté',
        'dos_dane': '🪵 Dos d’âne détecté',
        'trottoir': '📦 Choc trottoir détecté',
        'nid_de_poule': '🚧 Nid de poule détecté',
        'stop': '⏸️ Stop détecté',
        'wait': '⏱️ Wait détecté',
    }

    for key, label in mapping.items():
        sig_detected = normalize_detection_result(detected_signatures.get(key, False))
        realism_detected = normalize_detection_result(realism.get(label, False))
        assert sig_detected == realism_detected, f"Incohérence pour {key}: signature={sig_detected}, realism={realism_detected}"

    assert initial_detected, "L'accélération initiale n'a pas été détectée immédiatement après injection."
    assert final_detected, "La décélération finale n'a pas été détectée immédiatement après injection."

    print("✅ Test pipeline cohérent : toutes les signatures sont bien présentes et labellisées.")

if __name__ == "__main__":
    test_pipeline_consistency()

import yaml
from datetime import datetime

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

CONFIG = {
    "injection": {
        "acceleration": {"amplitude": 5.0, "duration_s": 0.8, "min_speed_kmh": 30},
        "freinage": {"amplitude": -6.0, "duration_s": 0.8, "min_speed_kmh": 30},
        "dos_dane": {"total_points": 20, "amplitude_peak_positive": 7.6, "amplitude_peak_negative": -7.6},
        "trottoir": {"total_points": 13, "acc_y_mean": 1.75, "acc_z_mean": 8.0},
        "nid_de_poule": {"pattern_acc_z": [4.2, 3.5, -5.3, 6.1, 9.8, 9.8]},
    },
    "injection_indices": [100, 42265, 84431, 126597, 168763],
    "general": {
        "force_speed_before_injection": 30,
        "inertial_noise": True,
        "detection_thresholds": {
            "acc_y_virage": 0.5,
            "heading_virage": 15,
            "validate_turns_threshold": 0.15  # Seuil recommandé ajusté ici
        }
    }
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
config_file = f"config_simulation_{timestamp}.yaml"

with open(config_file, 'w') as f:
    yaml.dump(CONFIG, f, sort_keys=False)

print(f"✅ Configuration sauvegardée dans {config_file}")

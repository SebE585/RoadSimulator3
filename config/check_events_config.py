

import yaml
import os

REQUIRED_FIELDS = {
    "acceleration": ["max_events", "acc_x_mean", "acc_x_std", "gyro_axes_used"],
    "freinage": ["max_events", "acc_x_mean", "acc_x_std", "gyro_axes_used"],
    "nid_de_poule": ["max_events", "pattern", "gyro_axes_used", "gyro_x_mean", "gyro_x_std"],
    "dos_dane": ["max_events", "phase_length", "amplitude_step", "gyro_axes_used", "gyro_x_mean", "gyro_x_std"],
    "trottoir": ["max_events", "duration_pts", "acc_y_mean", "acc_z_mean", "gyro_axes_used"],
    "stop": ["duration_pts", "max_events", "gyro_axes_used"],
    "wait": ["duration_pts", "max_events", "gyro_axes_used"],
    "initial": ["duration_pts", "v_max_kmh", "acc_x"],
    "final": ["duration_pts", "v_final_kmh", "acc_x"],
    "ouverture": ["max_events", "duration_pts", "gyro_y_mean", "gyro_y_std", "gyro_z_mean", "gyro_z_std"]
}

def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def validate_event_blocks(config):
    errors = []
    for event, fields in REQUIRED_FIELDS.items():
        if event not in config:
            errors.append(f"[MISSING BLOCK] {event}")
            continue
        for field in fields:
            if field not in config[event]:
                errors.append(f"[MISSING FIELD] {event}.{field}")
    return errors

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "events.yaml")
    config = load_yaml_config(config_path)
    errors = validate_event_blocks(config)
    if errors:
        print("❌ Problèmes détectés dans events.yaml :")
        for e in errors:
            print(" -", e)
    else:
        print("✅ events.yaml complet et valide.")
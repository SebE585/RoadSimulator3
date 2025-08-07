import json
import os
from core.utils import get_simulation_output_dir

def generate_summary_json(summary_dict, output_path):
    """
    Écrit le résumé de simulation au format JSON.
    """
    import pandas as pd
    import numpy as np

    def make_serializable(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        return obj

    summary_dict = {str(k): make_serializable(v) for k, v in summary_dict.items()}
    with open(output_path, "w") as f:
        json.dump(summary_dict, f, indent=2, ensure_ascii=False, default=str)

def generate_summary_log(summary_dict, output_path):
    """
    Écrit le résumé de simulation au format texte lisible.
    """
    with open(output_path, "w") as f:
        for key, value in summary_dict.items():
            f.write(f"{key} : {value}\n")

import os
import pandas as pd
from simulator.vizualisation.generate_all_outputs import generate_all_outputs_from_csv
from core.utils import get_simulation_output_dir
from datetime import datetime


def main():
    input_csv = "output_simulated_trajectory.csv"
    if not os.path.exists(input_csv):
        print(f"[ERROR] Fichier CSV introuvable : {input_csv}")
        return

    print(f"[INFO] 📄 Génération des outputs depuis : {input_csv}")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = get_simulation_output_dir(timestamp)
    generate_all_outputs_from_csv(input_csv, output_dir=output_dir)
    print(f"[INFO] ✅ Génération terminée.")

if __name__ == "__main__":
    main()
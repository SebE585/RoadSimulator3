import matplotlib.pyplot as plt
import pandas as pd
import os

def generate_speed_graph(df, output_path):
    """
    Génère un graphique de la vitesse au cours du temps.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(pd.to_datetime(df["timestamp"]), df["speed"], label="Vitesse (km/h)", color="blue")
    plt.xlabel("Temps")
    plt.ylabel("Vitesse (km/h)")
    plt.title("Évolution de la vitesse")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_stop_graph(df, output_path):
    """
    Génère un graphique indiquant les phases de stop et wait.
    """
    plt.figure(figsize=(12, 2))
    events = df["event"].fillna("")
    stop_mask = events.str.contains("stop")
    wait_mask = events.str.contains("wait")

    plt.plot(df["timestamp"], stop_mask * 1.0, label="Stop", color="red", alpha=0.5)
    plt.plot(df["timestamp"], wait_mask * 1.0, label="Wait", color="orange", alpha=0.5)

    plt.yticks([0, 1], ["", "Présence"])
    plt.xlabel("Temps")
    plt.title("Phases de Stop et Wait")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

import matplotlib.pyplot as plt

@deprecated
def plot_speed(df):
    logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
    plt.figure(figsize=(10,4))
    plt.plot(df['timestamp'], df['speed'])
    plt.xlabel('Timestamp')
    plt.ylabel('Speed (km/h)')
    plt.title('Vitesse vs Temps')
    plt.grid(True)
    plt.show()
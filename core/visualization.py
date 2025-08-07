import matplotlib.pyplot as plt

def plot_speed(df):
    plt.figure(figsize=(10,4))
    plt.plot(df['timestamp'], df['speed'])
    plt.xlabel('Timestamp')
    plt.ylabel('Speed (km/h)')
    plt.title('Vitesse vs Temps')
    plt.grid(True)
    plt.show()
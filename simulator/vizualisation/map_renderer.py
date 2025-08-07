import folium
from folium.plugins import BeautifyIcon
from folium.plugins import MarkerCluster
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import os
import pandas as pd

def generate_html_map(df, output_html_path):
    """
    Génère une carte HTML avec tracé de la trajectoire et affichage des événements.
    """
    m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=13)

    # Tracé de la trajectoire
    points = df[["lat", "lon"]].dropna().values.tolist()
    folium.PolyLine(points, color="blue", weight=3).add_to(m)

    # Marqueurs d’événements
    if "event" in df.columns:
        events_df = df[df["event"].notna()][["lat", "lon", "event"]].copy()
        events_df["lat_rounded"] = events_df["lat"].round(5)
        events_df["lon_rounded"] = events_df["lon"].round(5)
        events_df["event_key"] = events_df["event"] + "_" + events_df["lat_rounded"].astype(str) + "_" + events_df["lon_rounded"].astype(str)
        events_df = events_df.drop_duplicates(subset="event_key")

        max_per_type = 30
        sampled_events = []
        for evt_type, group in events_df.groupby("event"):
            sampled_group = group.iloc[::max(1, len(group) // max_per_type)]
            sampled_events.append(sampled_group.head(max_per_type))
        sampled_df = pd.concat(sampled_events)

        marker_cluster = MarkerCluster(maxClusterRadius=50).add_to(m)
        for _, row in sampled_df.iterrows():
            folium.Marker(
                location=[row["lat"], row["lon"]],
                icon=BeautifyIcon(
                    icon_shape='marker',
                    border_color='red',
                    text_color='white',
                    number=1
                ),
                popup=row["event"]
            ).add_to(marker_cluster)

    m.save(output_html_path)

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def export_png_map(input_html_path, output_png_path, wait_time=10):

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1200,800')

    driver = webdriver.Chrome(options=options)
    driver.get("file://" + os.path.abspath(input_html_path))

    try:
        WebDriverWait(driver, wait_time).until(
            lambda d: d.execute_script('return typeof L !== "undefined" && document.readyState === "complete";')
        )
        time.sleep(2)  # laisser le temps au rendu Leaflet
        driver.save_screenshot(output_png_path)
    except Exception as e:
        print(f"[ERROR] Timeout ou erreur lors du rendu : {e}")
    finally:
        driver.quit()
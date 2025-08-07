import requests

API_URL = "http://127.0.0.1:5002/nearest_road_batch"

def test_nearest_roads_batch():
    points = [
        {"lat": 49.4431, "lon": 1.0993},  # Rouen
        {"lat": 49.4944, "lon": 0.1079},  # Le Havre
        {"lat": 49.1829, "lon": -0.3700}, # Caen
        {"lat": 49.0266, "lon": 1.1508},  # Évreux
        {"lat": 49.9222, "lon": 1.0778},  # Dieppe
        {"lat": 48.4333, "lon": 0.0833},  # Alençon
        {"lat": 49.2559, "lon": 0.3928},
        {"lat": 49.6383, "lon": 0.2239},
        {"lat": 48.7644, "lon": 0.8714},
        {"lat": 49.4876, "lon": 0.7656},
        {"lat": 49.1811, "lon": 1.2072},
        {"lat": 49.1455, "lon": -1.1000},
        {"lat": 48.7850, "lon": 0.5600},
        {"lat": 49.3854, "lon": 1.0186},
        {"lat": 49.6290, "lon": 1.0204},
        {"lat": 49.5877, "lon": 0.8617},
        {"lat": 49.2100, "lon": 0.3641},
        {"lat": 49.7206, "lon": 1.1500},
        {"lat": 48.7916, "lon": 0.4171},
        {"lat": 48.8543, "lon": -0.5730},
        {"lat": 49.5586, "lon": 0.8400},
        {"lat": 49.7094, "lon": 1.3956},
        {"lat": 49.1815, "lon": 1.0097},
        {"lat": 49.0964, "lon": 1.5102},
        {"lat": 49.4184, "lon": 0.9613},
        {"lat": 49.2721, "lon": 0.7000},
        {"lat": 48.9087, "lon": 0.1840},
        {"lat": 49.4044, "lon": 1.0303},
        {"lat": 48.9181, "lon": -0.1881},
        {"lat": 49.2014, "lon": 0.9330},
        {"lat": 49.4273, "lon": 1.0704},
        {"lat": 48.9276, "lon": 0.7795},
        {"lat": 49.1783, "lon": 0.5639},
        {"lat": 49.0281, "lon": 0.6590},
        {"lat": 49.1621, "lon": -0.3094},
        {"lat": 48.9004, "lon": 0.6019},
        {"lat": 48.7565, "lon": 0.0428},
        {"lat": 49.4224, "lon": 0.1043},
        {"lat": 48.8615, "lon": 0.5069},
        {"lat": 49.5006, "lon": 0.6641},
        {"lat": 49.5383, "lon": 0.2186},
        {"lat": 48.9505, "lon": -0.3657},
        {"lat": 49.1166, "lon": 0.3544},
        {"lat": 49.3439, "lon": 1.1030},
        {"lat": 49.0778, "lon": 1.4690},
        {"lat": 49.0084, "lon": 0.6274},
        {"lat": 48.8553, "lon": 0.9241},
        {"lat": 49.0500, "lon": 0.4958},
        {"lat": 49.4839, "lon": 1.1272},
        {"lat": 49.2093, "lon": 1.2938},
    ]
    buffer_m = 100

    try:
        response = requests.post(API_URL, json=points, params={'buffer_m': buffer_m})
        response.raise_for_status()
        data = response.json()

        print("Résultats du batch:")
        for idx, result in enumerate(data):
            print(f"Point {idx+1}: lat={points[idx]['lat']}, lon={points[idx]['lon']}")
            print(f" -> nearest_node: {result.get('nearest_node')}")
            print(f" -> road_type: {result.get('road_type')}")
            print(f" -> maxspeed: {result.get('maxspeed')}")
            print(f" -> nearest_distance_m: {result.get('nearest_distance_m', 'N/A')}")
            print(f" -> candidates_count: {result.get('candidates_count', 'N/A')}")
            print("---")
    except Exception as e:
        print(f"[ERROR] Échec de l'appel API batch: {e}")

if __name__ == "__main__":
    test_nearest_roads_batch()

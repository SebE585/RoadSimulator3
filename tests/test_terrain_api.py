import requests

def test_srtm_api_with_normandy_coordinates():
    """
    Teste l'API SRTM locale sur 3 points GPS en Normandie (autour de Rouen).
    """
    url = "http://localhost:5004/enrich_terrain"
    payload = [
        {"lat": 49.4401, "lon": 1.0931},     # Rouen
        {"lat": 49.3925, "lon": 1.1361},     # Darnétal
        {"lat": 49.3710, "lon": 1.1486},     # Romilly-sur-Andelle
    ]

    response = requests.post(url, json=payload)
    print("Status code:", response.status_code)
    print("Réponse texte:", response.text)

    assert response.status_code == 200, f"API status code != 200: {response.status_code}"

    data = response.json()
    assert isinstance(data, list), "La réponse n'est pas une liste"
    assert len(data) == 3, f"La réponse contient {len(data)} points au lieu de 3"

    for point in data:
        assert "altitude" in point, "Clé 'altitude' absente"
        assert "altitude_smoothed" in point, "Clé 'altitude_smoothed' absente"
        assert "slope_percent" in point, "Clé 'slope_percent' absente"
        assert isinstance(point["altitude"], (int, float)), "Altitude invalide"

    print("[✅ TEST PASSÉ] API SRTM répond correctement sur 3 points en Normandie.")


if __name__ == "__main__":
    test_srtm_api_with_normandy_coordinates()

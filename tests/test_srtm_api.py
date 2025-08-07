import requests

url = 'http://localhost:5004/enrich_terrain'
payload = [
    {"lat": 49.4, "lon": 1.1},
    {"lat": 49.5, "lon": 1.2},
    {"lat": 49.6, "lon": 1.3}
]

response = requests.post(url, json=payload)

print("Status code:", response.status_code)
print("Response JSON:")
print(response.json())

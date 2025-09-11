# Schéma du dataset RS3

Chaque simulation produit un dataset tabulaire enrichi.

---

## Colonnes principales (v1.0)
| Nom        | Type     | Unité   | Description |
|------------|----------|---------|-------------|
| timestamp  | datetime | s (UTC) | Échantillonnage 10 Hz |
| lat        | float64  | degré   | Latitude WGS84 |
| lon        | float64  | degré   | Longitude WGS84 |
| speed      | float32  | m/s     | Vitesse au sol |
| acc_x      | float32  | m/s²    | Accélération longitudinale |
| acc_y      | float32  | m/s²    | Accélération latérale |
| acc_z      | float32  | m/s²    | Accélération verticale |
| gyro_x     | float32  | rad/s   | Rotation X (roulis) |
| gyro_y     | float32  | rad/s   | Rotation Y (tangage) |
| gyro_z     | float32  | rad/s   | Rotation Z (lacet) |
| event      | string   | -       | Événement injecté (stop, virage, …) |

---

## Exemple CSV (extrait)
```csv
timestamp,lat,lon,speed,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,event
2025-09-06T08:00:00Z,48.8566,2.3522,0.0,0.01,-0.02,9.81,0.00,0.00,0.00,stop
2025-09-06T08:00:00.1Z,48.85661,2.35221,1.2,0.05,0.01,9.81,0.00,0.00,0.02,move
```

---

## Validations appliquées
- Ordre strict des timestamps
- Cohérence lat/lon
- Distribution bruit inertiel réaliste
- Colonnes gyro systématiques

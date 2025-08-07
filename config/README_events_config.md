
# 📄 Fichier de configuration des événements inertiels (`events.yaml`)

Le fichier `events.yaml` définit les paramètres expérimentaux de simulation liés à l’injection d’événements inertiels, d’arrêts moteurs (`stop`, `wait`), ainsi qu’aux phases transitoires du démarrage et de l’arrêt du véhicule.

Chaque section correspond à une famille d’événements ou de règles simulées dans le pipeline.

---

## ⚙️ Section `general`

| Clé | Description |
|-----|-------------|
| `hz` | Fréquence d’échantillonnage en Hz (typiquement 10 Hz). |
| `g` | Valeur de référence de la gravité, utilisée pour simuler les accélérations verticales. |
| `logging_level` | Niveau de journalisation (`DEBUG`, `INFO`, `WARNING`), pour contrôler le niveau de verbosité des logs. |

---

## ⚡ Événements ponctuels inertiels

### `acceleration`

Injection d’accélérations longitudinales réalistes.

| Clé | Description |
|-----|-------------|
| `max_events` | Nombre maximal d’événements à injecter. |
| `amplitude` | Valeur de `acc_x` injectée (en m/s²). |
| `duration_s` | Durée typique de l’accélération. |
| `max_attempts` | Nombre maximum de tentatives pour trouver une position valide. |

### `freinage`

Simulation d’un freinage brusque avec perte de vitesse.

| Clé | Description |
|-----|-------------|
| `min_delta_kmh` / `max_delta_kmh` | Intervalle de perte de vitesse ciblée. |
| `acc_x_start` | Accélération initiale négative. |
| `max_events`, `max_attempts` | Idem `acceleration`. |

### `nid_de_poule`, `dos_dane`, `trottoir`

Injection de profils caractéristiques sur l’axe vertical et latéral (`acc_z`, `acc_y`), selon les spécifications physiques définies par les signatures inertiels.

---

## 🛑 Événements prolongés : arrêts moteur ou temporisations

### `stop`

Représente un arrêt moteur (≥ 2 minutes) avec profil d’arrêt complet.

| Clé | Description |
|-----|-------------|
| `min_duration_s` | Durée minimale d’un arrêt. |
| `speed_threshold_kmh` | Seuil de vitesse pour arrêt effectif. |
| `max_events` | Nombre maximal de stops simulés. |

### `wait`

Arrêt moteur tournant (ex : livraisons courtes, ≤ 2 minutes).

| Clé | Description |
|-----|-------------|
| `min_duration_s` / `max_duration_s` | Durée de l’arrêt. |
| `max_events` | Nombre de waits autorisés. |

---

## 🚗 Phases de transition

### `initial`

Phase d’accélération initiale au démarrage du véhicule.

| Clé | Description |
|-----|-------------|
| `v_max_kmh` | Vitesse cible à atteindre. |
| `duration_s` | Durée d'accélération. |
| `acc_x` | Accélération longitudinale constante. |

### `final`

Décélération progressive à l’approche de la fin du parcours.

| Clé | Description |
|-----|-------------|
| `duration_s` | Durée de freinage final. |
| `v_final_kmh` | Vitesse cible à atteindre (souvent 0). |

---

## 🔄 Surcharge dynamique dans un script Python

Il est possible d’outrepasser dynamiquement une valeur YAML dans un script Python :

```python
from simulator.events.config import CONFIG

CONFIG["freinage"]["max_events"] = 2
CONFIG["general"]["logging_level"] = "DEBUG"

from simulator.events import apply_all_events
df = apply_all_events(df)
```

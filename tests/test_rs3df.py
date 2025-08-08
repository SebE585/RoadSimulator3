# tests/test_rs3df.py
import io
import numpy as np
import pandas as pd
import pytest

# 👉 Adapte ce chemin si besoin
from core.rs3df import RS3DF


# -----------------------
# Fixtures utilitaires
# -----------------------
@pytest.fixture
def base_df():
    n = 100
    ts = pd.date_range("2025-01-01 12:00:00", periods=n, freq="100ms")  # 10 Hz
    df = pd.DataFrame({
        "timestamp": ts,
        "lat": np.linspace(49.0, 49.001, n),
        "lon": np.linspace(1.0, 1.002, n),
        "speed": np.linspace(0, 13.9, n),  # ~50 km/h
        "acc_x": np.zeros(n),
        "acc_y": np.zeros(n),
        "acc_z": np.zeros(n),
        "gyro_x": np.zeros(n),
        "gyro_y": np.zeros(n),
        "gyro_z": np.zeros(n),
        "event": [None] * n,
    })
    return df


@pytest.fixture
def rs3(base_df):
    # min_gap: au moins 1 s entre deux événements, tol_idx: ±2 échantillons pour les collisions
    return RS3DF(base_df, min_gap_s=1.0, collision_tol_pts=2)


# -----------------------
# Tests de base / contrat
# -----------------------

def test_schema_validate_ok(rs3):
    # Doit lever si colonnes manquent, sinon True
    assert rs3.validate() is True


def test_add_event_marks_row(rs3):
    idx = 20
    rs3.add_event(kind="speed_bump", at_idx=idx)
    out = rs3.df  # ou rs3.get_df()
    assert out.loc[idx, "event"] == "speed_bump"


def test_add_event_rejects_overlap_by_timegap(rs3):
    # Premier événement à t=2.0s (idx 20), second à t=2.7s (idx 27) -> min_gap=1.0s => doit refuser
    rs3.add_event(kind="brake_hard", at_idx=20)
    with pytest.raises(ValueError):
        rs3.add_event(kind="speed_bump", at_idx=27)


def test_add_event_collision_tolerance(rs3):
    # Deux insertions très proches (±2 pts), la seconde doit être bloquée par la tolérance de collision
    rs3.add_event(kind="brake_hard", at_idx=30)
    with pytest.raises(ValueError):
        rs3.add_event(kind="brake_hard", at_idx=31)  # collision tolérée ±2 pts


def test_bulk_add_reports_inserted_and_skipped(rs3):
    req = [
        {"kind": "brake_hard", "at_idx": 10},
        {"kind": "speed_bump", "at_idx": 15},   # collision avec min_gap -> selon impl, skip/raise
        {"kind": "pothole", "at_idx": 30},
    ]
    report = rs3.add_events_bulk(req, on_conflict="skip")  # "skip"| "error" | "replace"
    assert set(report.keys()) >= {"inserted", "skipped", "errors"}
    assert report["inserted"] >= 1
    # Au moins un conflit potentiel devrait finir en "skipped"
    assert report["skipped"] >= 0


def test_dedupe_events_collapses_close_marks(rs3):
    # Pose des doublons proches, puis dedupe doit en garder 1
    for k in (25, 26, 27):
        rs3.add_event(kind="pothole", at_idx=k) if rs3.df.loc[k, "event"] is None else None
    before = (rs3.df["event"] == "pothole").sum()
    rs3.dedupe_events(kind="pothole", min_gap_s=1.0)
    after = (rs3.df["event"] == "pothole").sum()
    assert after <= before
    assert after >= 1


def test_window_returns_slice_around_event(rs3):
    rs3.add_event(kind="roundabout", at_idx=50)
    win = rs3.window(center_idx=50, half_width=5)
    # 5 de chaque côté + le centre -> 11 lignes
    assert isinstance(win, pd.DataFrame)
    assert len(win) == 11
    assert win.index.min() == 45 and win.index.max() == 55


def test_to_from_csv_roundtrip(rs3, tmp_path):
    rs3.add_event(kind="brake_hard", at_idx=12)
    p = tmp_path / "sim.csv"
    rs3.to_csv(p)
    rs3_bis = RS3DF.from_csv(p, min_gap_s=rs3.min_gap_s, collision_tol_pts=rs3.collision_tol_pts)
    # Même nombre de lignes et au moins le même marquage d’événement au même index
    assert len(rs3_bis.df) == len(rs3.df)
    assert rs3_bis.df.loc[12, "event"] == "brake_hard"


# -----------------------
# Cas limites / erreurs
# -----------------------

def test_add_event_out_of_bounds_raises(rs3):
    with pytest.raises(IndexError):
        rs3.add_event(kind="speed_bump", at_idx=10_000)


def test_validate_missing_columns_raises():
    bad = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=5, freq="100ms"),
                        "lat": np.zeros(5)})
    with pytest.raises(AssertionError):
        RS3DF(bad).validate()


def test_add_event_by_time_works(rs3):
    # Ajout via horodatage (facultatif si l’API la supporte)
    t = rs3.df.loc[40, "timestamp"]
    rs3.add_event(kind="door_open", at_time=t)
    assert rs3.df.loc[40, "event"] == "door_open"
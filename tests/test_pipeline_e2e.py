"""
Test E2E du pipeline RS3 — vérifie que chaque stage produit le bon résultat.
Exécuter : python tests/test_pipeline_e2e.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from engine.context import Context
from engine.stages.imu_projector import IMUProjector
from engine.stages.noise_injector import NoiseInjector
from engine.stages.device_rotator import DeviceRotator
from engine.stages.events_tagger import EventsTagger
from engine.stages.event_injector import EventInjector
from engine.stages.multi_rate_sampler import MultiRateSampler
from engine.stages.exporter import _load_schema, _apply_schema


def _make_fake_ctx():
    """Crée un contexte avec un DataFrame minimal post-SpeedSync (avant IMUProjector)."""
    n = 2000
    hz = 10
    dt = 1.0 / hz
    t = pd.date_range("2025-01-01T08:00:00Z", periods=n, freq=f"{int(1000/hz)}ms", tz="UTC")

    # stop 200pts → accel 200pts → cruise 1200pts → brake 200pts → stop 200pts
    v = np.concatenate([
        np.zeros(200),
        np.linspace(0, 12, 200),
        np.full(1200, 12.0),
        np.linspace(12, 0, 200),
        np.zeros(200),
    ])

    lat = 49.33 + np.cumsum(v * dt / 111_000)
    lon = np.full(n, 1.38)

    df = pd.DataFrame({
        "timestamp": t.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "lat": lat, "lon": lon,
        "speed": v,
        "flag_stop": np.where(v < 0.1, 1, 0).astype(float),
        "flag_wait": np.zeros(n),
    })

    ctx = Context(cfg={"sim": {"hz": hz}})
    ctx.df = df
    ctx.set_meta("hz", hz)
    return ctx


def test_imu_projector():
    print("▶ IMUProjector")
    ctx = _make_fake_ctx()
    IMUProjector().run(ctx)
    assert "acc_x" in ctx.df.columns, "acc_x manquant"
    assert "acc_z" in ctx.df.columns, "acc_z manquant"
    assert "gyro_z" in ctx.df.columns, "gyro_z manquant"
    # Au repos, acc_z ≈ G
    still = ctx.df[ctx.df["speed"] < 0.3]
    az_mean = still["acc_z"].mean()
    assert abs(az_mean - 9.81) < 0.5, f"acc_z au repos = {az_mean:.2f}, attendu ~9.81"
    print(f"  acc_z au repos = {az_mean:.3f} ✅")
    return ctx


def test_noise_injector(ctx):
    print("▶ NoiseInjector")
    before_ax = ctx.df["acc_x"].copy()
    NoiseInjector(sigma_acc=0.03, sigma_gyro=0.002).run(ctx)
    diff = (ctx.df["acc_x"] - before_ax).std()
    assert diff > 0.01, f"Bruit non ajouté (diff std = {diff})"
    print(f"  bruit ajouté: sigma_diff = {diff:.4f} ✅")
    return ctx


def test_device_rotator(ctx):
    print("▶ DeviceRotator (15°, 8°, 20°)")
    still_before = ctx.df[ctx.df["speed"] < 0.3]["acc_x"].mean()
    DeviceRotator(roll_deg=15.0, pitch_deg=8.0, yaw_deg=20.0).run(ctx)
    still_after = ctx.df[ctx.df["speed"] < 0.3]["acc_x"].mean()
    print(f"  acc_x au repos: avant={still_before:.3f}, après={still_after:.3f}")
    assert abs(still_after) > 1.0, f"Rotation pas appliquée: acc_x = {still_after:.3f}"
    assert "device_rotation_deg" in ctx.meta, "Meta rotation manquante"
    print(f"  acc_x décalé de {still_after:.3f} m/s² ✅")
    return ctx


def test_event_injector(ctx):
    print("▶ EventInjector")
    EventsTagger().run(ctx)  # nécessaire avant EventInjector
    EventInjector(n_harsh_brake=2, n_pothole=1, n_door_open=1, seed=42).run(ctx)
    events = ctx.df["event"].value_counts().to_dict()
    print(f"  events: {events}")
    assert "harsh_brake" in events or "door_open" in events, f"Aucun événement injecté: {events}"
    print(f"  événements injectés ✅")
    return ctx


def test_multi_rate_sampler(ctx):
    print("▶ MultiRateSampler (GPS 1Hz, gyro off)")
    MultiRateSampler(gps_hz=1.0, imu_hz=10.0, gyro_enabled=False).run(ctx)
    lat_nan = ctx.df["lat"].isna().sum()
    speed_nan = ctx.df["speed"].isna().sum()
    gyro_nan = ctx.df["gyro_z"].isna().sum()
    n = len(ctx.df)
    print(f"  lat NaN: {lat_nan}/{n}")
    print(f"  speed NaN: {speed_nan}/{n}")
    print(f"  gyro_z NaN: {gyro_nan}/{n}")
    assert lat_nan > n * 0.8, f"GPS pas downsamplé: lat NaN = {lat_nan}/{n}"
    assert speed_nan > n * 0.8, f"Speed pas downsamplé: speed NaN = {speed_nan}/{n}"
    assert gyro_nan == n, f"Gyro pas désactivé: NaN = {gyro_nan}/{n}"
    # acc_x doit être intact (pas NaN)
    acc_nan = ctx.df["acc_x"].isna().sum()
    assert acc_nan == 0, f"acc_x ne devrait pas avoir de NaN: {acc_nan}"
    print(f"  acc_x intact (0 NaN) ✅")
    # Rotation doit être préservée
    still = ctx.df[(ctx.df["speed"].fillna(0) < 0.3) & ctx.df["acc_x"].notna()]
    if len(still) > 5:
        ax_still = still["acc_x"].mean()
        print(f"  acc_x au repos après sampler: {ax_still:.3f} (doit être ~2.1)")
        assert abs(ax_still) > 1.0, f"Rotation perdue après sampler: acc_x = {ax_still}"
    print(f"  MultiRateSampler OK ✅")
    return ctx


def test_schema_preserves_nan(ctx):
    print("▶ Schema _apply_schema (NaN preservation)")
    schema = _load_schema({})
    if not schema:
        print("  ⏭ Pas de schema trouvé — skip")
        return ctx

    df_out = _apply_schema(ctx.df, schema)
    lat_nan = df_out["lat"].isna().sum()
    gyro_nan = df_out["gyro_z"].isna().sum() if "gyro_z" in df_out.columns else 0
    acc_nan = df_out["acc_x"].isna().sum()
    n = len(df_out)

    print(f"  lat NaN après schema: {lat_nan}/{n}")
    print(f"  gyro_z NaN après schema: {gyro_nan}/{n}")
    print(f"  acc_x NaN après schema: {acc_nan}/{n}")

    assert lat_nan > n * 0.5, f"Schema a rempli lat NaN: {lat_nan}/{n}"
    assert acc_nan == 0, f"Schema a introduit des NaN dans acc_x: {acc_nan}"

    # Vérifier rotation préservée
    still = df_out[(df_out["speed"].fillna(0) < 0.3) & df_out["acc_x"].notna()]
    if len(still) > 5:
        ax = still["acc_x"].mean()
        print(f"  acc_x au repos dans CSV final: {ax:.3f}")
        assert abs(ax) > 1.0, f"Rotation perdue dans le CSV: acc_x = {ax}"

    print(f"  Schema OK — NaN préservés ✅")
    return ctx


if __name__ == "__main__":
    print("=" * 60)
    print("TEST E2E Pipeline RS3")
    print("=" * 60)

    passed, failed = 0, 0
    tests = [
        ("IMUProjector", lambda: test_imu_projector()),
    ]

    # Chaîne séquentielle (chaque test passe le ctx au suivant)
    ctx = None
    chain = [
        ("IMUProjector", test_imu_projector),
        ("NoiseInjector", lambda c=None: test_noise_injector(ctx)),
        ("DeviceRotator", lambda c=None: test_device_rotator(ctx)),
        ("EventInjector", lambda c=None: test_event_injector(ctx)),
        ("MultiRateSampler", lambda c=None: test_multi_rate_sampler(ctx)),
        ("Schema", lambda c=None: test_schema_preserves_nan(ctx)),
    ]

    # Exécuter séquentiellement
    ctx = _make_fake_ctx()
    for name, fn in [
        ("IMUProjector", lambda: test_imu_projector()),
    ]:
        pass  # handled below

    # Run the chain properly
    try:
        ctx = test_imu_projector()
        passed += 1
    except AssertionError as e:
        print(f"  ❌ {e}"); failed += 1; ctx = None

    if ctx:
        try:
            ctx = test_noise_injector(ctx)
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {e}"); failed += 1

    if ctx:
        try:
            ctx = test_device_rotator(ctx)
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {e}"); failed += 1

    if ctx:
        try:
            ctx = test_event_injector(ctx)
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {e}"); failed += 1

    if ctx:
        try:
            ctx = test_multi_rate_sampler(ctx)
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {e}"); failed += 1

    if ctx:
        try:
            ctx = test_schema_preserves_nan(ctx)
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {e}"); failed += 1

    print(f"\n{'=' * 60}")
    print(f"Résultat: {passed}/{passed + failed} tests passés")
    if failed:
        sys.exit(1)
    else:
        print("Tous les tests passent ✅")

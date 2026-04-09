"""
RoadSimulator3 — Web Simulator v2.0
Réécriture propre — session refonte UI 25 mars 2026
"""
from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import folium
import folium.plugins
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from streamlit_folium import st_folium

# ── i18n (self-contained) ────────────────────────────────────────────────────

_RS3_TR: dict[str, dict[str, str]] = {
    "fr": {
        "rs3.subtitle":       "Simulateur télématique — Placez des arrêts, configurez, lancez",
        "rs3.stops":          "Arrêts",
        "rs3.click_hint":     "Cliquez sur la carte pour ajouter le DEPOT puis les clients.",
        "rs3.freq":           "Fréquence",
        "rs3.gps_noise":      "Bruit GPS réaliste",
        "rs3.blackouts":      "Blackouts tunnel",
        "rs3.cold_drift":     "Drift démarrage (m)",
        "rs3.imu_title":      "IMU (Accéléromètre + Gyroscope)",
        "rs3.acc_noise":      "Bruit acc (m/s²)",
        "rs3.gyro_on":        "Gyroscope activé",
        "rs3.orientation":    "Orientation du boîtier",
        "rs3.events_title":   "Événements à injecter",
        "rs3.brakes":         "Freinages",
        "rs3.accels":         "Accélérations",
        "rs3.bumps":          "Dos d'âne",
        "rs3.potholes":       "Nids de poule",
        "rs3.turns":          "Virages",
        "rs3.door":           "Ouv. porte",
        "rs3.carto":          "Services carto",
        "rs3.launch":         "🚀 Lancer la simulation",
        "rs3.running":        "Simulation RS3 en cours...",
        "rs3.success":        "Simulation terminée —",
        "rs3.fail":           "Échec —",
        "rs3.error":          "Erreur:",
        "rs3.btn_depot":      "Retour DEPOT",
        "rs3.btn_del_last":   "Supprimer dernier",
        "rs3.btn_clear":      "Effacer tout",
        "rs3.tab_map":        "Carte",
        "rs3.tab_sensors":    "Capteurs",
        "rs3.tab_severity":   "Severity",
        "rs3.tab_quality":    "Qualité",
        "rs3.chart_accel":    "Accéléromètre",
        "rs3.chart_gyro":     "Gyroscope",
        "rs3.gyro_off":       "Gyroscope désactivé",
        "rs3.lang":           "Langue",
        "rs3.v_median":       "V médiane",
        "rs3.acc_std":        "σ acc_x",
        "rs3.gyro_std":       "σ gyro_z",
        "rs3.hz_obs":         "Hz obs",
    },
    "en": {
        "rs3.subtitle":       "Telematic simulator — Place stops, configure, launch",
        "rs3.stops":          "Stops",
        "rs3.click_hint":     "Click on the map to add the DEPOT then customers.",
        "rs3.freq":           "Frequency",
        "rs3.gps_noise":      "Realistic GPS noise",
        "rs3.blackouts":      "Tunnel blackouts",
        "rs3.cold_drift":     "Cold start drift (m)",
        "rs3.imu_title":      "IMU (Accelerometer + Gyroscope)",
        "rs3.acc_noise":      "Acc noise (m/s²)",
        "rs3.gyro_on":        "Gyroscope enabled",
        "rs3.orientation":    "Device orientation",
        "rs3.events_title":   "Events to inject",
        "rs3.brakes":         "Hard brakes",
        "rs3.accels":         "Hard accels",
        "rs3.bumps":          "Speed bumps",
        "rs3.potholes":       "Potholes",
        "rs3.turns":          "Sharp turns",
        "rs3.door":           "Door open",
        "rs3.carto":          "Map services",
        "rs3.launch":         "🚀 Launch simulation",
        "rs3.running":        "RS3 simulation running...",
        "rs3.success":        "Simulation complete —",
        "rs3.fail":           "Failed —",
        "rs3.error":          "Error:",
        "rs3.btn_depot":      "Return to DEPOT",
        "rs3.btn_del_last":   "Delete last",
        "rs3.btn_clear":      "Clear all",
        "rs3.tab_map":        "Map",
        "rs3.tab_sensors":    "Sensors",
        "rs3.tab_severity":   "Severity",
        "rs3.tab_quality":    "Quality",
        "rs3.chart_accel":    "Accelerometer",
        "rs3.chart_gyro":     "Gyroscope",
        "rs3.gyro_off":       "Gyroscope disabled",
        "rs3.lang":           "Language",
        "rs3.v_median":       "V median",
        "rs3.acc_std":        "σ acc_x",
        "rs3.gyro_std":       "σ gyro_z",
        "rs3.hz_obs":         "Hz obs",
    },
}

def _rs3_lang() -> str:
    return st.session_state.get("rs3_lang", "fr")

def L(key: str) -> str:
    """Translate key using current RS3 language."""
    lang = _rs3_lang()
    return _RS3_TR.get(lang, _RS3_TR["fr"]).get(key, _RS3_TR["fr"].get(key, key))

# ── Setup ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="RS3 Simulator", page_icon="🚛", layout="wide", menu_items={})
st.markdown("""<style>
.stDeployButton,.stMainMenu,header[data-testid="stHeader"],footer,
div[data-testid="stStatusWidget"]{display:none!important}
div[data-testid="stMetricValue"]{color:#0066CC;font-weight:700}
</style>""", unsafe_allow_html=True)

DEFAULT_CENTER = [49.38, 1.25]
DEFAULT_ZOOM = 11
OSRM_URL = "http://localhost:5003"

for k, v in [("stops", []), ("sim_ctx", None), ("sim_outdir", None), ("sim_df", None), ("last_click", None), ("rs3_lang", "fr")]:
    if k not in st.session_state:
        st.session_state[k] = v

with st.sidebar:
    st.session_state["rs3_lang"] = st.radio(
        L("rs3.lang"), ["fr", "en"],
        index=0 if st.session_state.get("rs3_lang", "fr") == "fr" else 1,
        format_func=lambda x: "🇫🇷 Français" if x == "fr" else "🇬🇧 English",
        key="_rs3_lang_radio", horizontal=True,
    )

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown(
    "<div style='display:flex;align-items:center;gap:12px;margin-bottom:16px'>"
    "<span style='font-size:2.5em'>🚛</span>"
    "<div><h1 style='margin:0;font-size:1.8em'>RoadSimulator3</h1>"
    f"<p style='margin:0;color:#888'>{L('rs3.subtitle')}</p>"
    "</div></div>", unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# CARTE + CONFIG
# ══════════════════════════════════════════════════════════════════════════════

col_map, col_cfg = st.columns([3, 2])

with col_map:
    m = folium.Map(location=DEFAULT_CENTER, zoom_start=DEFAULT_ZOOM, tiles="OpenStreetMap")
    for i, s in enumerate(st.session_state.stops):
        color = "red" if s["id"] == "DEPOT" else "blue"
        folium.Marker([s["lat"], s["lon"]], popup=f"{s['id']} ({s['service_s']}s)",
                      icon=folium.Icon(color=color, icon="home" if s["id"] == "DEPOT" else "info-sign")).add_to(m)
    if len(st.session_state.stops) >= 2:
        folium.PolyLine([[s["lat"], s["lon"]] for s in st.session_state.stops],
                        color="green", weight=3, opacity=0.7).add_to(m)

    map_data = st_folium(m, width=700, height=450, returned_objects=["last_clicked"])

    # Add stop on click — NO st.rerun() to avoid iframe destruction
    if map_data and map_data.get("last_clicked"):
        lat = round(map_data["last_clicked"]["lat"], 6)
        lng = round(map_data["last_clicked"]["lng"], 6)
        if not any(abs(s["lat"] - lat) < 1e-5 and abs(s["lon"] - lng) < 1e-5 for s in st.session_state.stops):
            n = len(st.session_state.stops)
            sid = "DEPOT" if n == 0 else f"C{n}"
            st.session_state.stops.append({"id": sid, "lat": lat, "lon": lng, "service_s": 0 if sid == "DEPOT" else 30})
            # No st.rerun() — markers appear on next natural interaction

    bc = st.columns(3)
    if bc[0].button(L("rs3.btn_depot")) and st.session_state.stops:
        d = st.session_state.stops[0]
        st.session_state.stops.append({"id": "DEPOT", "lat": d["lat"], "lon": d["lon"], "service_s": 0})
        st.rerun()
    if bc[1].button(L("rs3.btn_del_last")) and st.session_state.stops:
        st.session_state.stops.pop()
        st.rerun()
    if bc[2].button(L("rs3.btn_clear")):
        st.session_state.stops = []
        st.session_state.sim_ctx = st.session_state.sim_df = st.session_state.sim_outdir = None
        st.rerun()

with col_cfg:
    st.markdown(f"**{L('rs3.stops')}**")
    if not st.session_state.stops:
        st.caption(L("rs3.click_hint"))
    for i, s in enumerate(st.session_state.stops):
        c1, c2 = st.columns([3, 1])
        c1.text(f"{s['id']} ({s['lat']:.4f}, {s['lon']:.4f})")
        st.session_state.stops[i]["service_s"] = c2.number_input(
            "s", min_value=0, value=s["service_s"], step=10, key=f"svc_{i}", label_visibility="collapsed")

    st.divider()

    # ── Device Profile ──────────────────────────
    _PROFILES = {
        "Custom": {},
        "Low-cost datalogger (no gyro)": {
            "gps_hz": 1, "imu_hz": 10, "sig_acc": 0.06, "gyro": False,
            "sig_pos": 2.0, "roll": 175.0, "pitch": 1.4, "yaw": 0.0,
            "desc": "Typical fleet datalogger (~€15), wired to vehicle battery. 50 Hz burst IMU, 1 Hz GPS, no gyro, gravity-compensated. Calibrated from field data (14 trips, France).",
        },
        "AEGIS SensorTile.Box": {
            "gps_hz": 1, "imu_hz": 10, "sig_acc": 0.02, "gyro": True,
            "sig_pos": 1.5, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            "desc": "ST SensorTile.Box (€50). 24 Hz IMU+gyro, raw gravity. Calibrated from AEGIS (33 trips, Austria).",
        },
        "iPhone (UAH-DriveSet)": {
            "gps_hz": 1, "imu_hz": 10, "sig_acc": 0.03, "gyro": True,
            "sig_pos": 3.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            "desc": "iPhone 6S (MPU-6500). 10 Hz IMU+gyro, gravity-compensated. Calibrated from UAH-DriveSet (40 trips, Spain).",
        },
        "Android Smartphone": {
            "gps_hz": 1, "imu_hz": 10, "sig_acc": 0.04, "gyro": True,
            "sig_pos": 4.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            "desc": "Typical Android phone. 10 Hz IMU+gyro, gravity-compensated, noisier GPS.",
        },
    }
    _pname = st.selectbox("📦 Device profile", list(_PROFILES.keys()), key="dev_profile")
    _prof = _PROFILES[_pname]
    if _prof.get("desc"):
        st.caption(_prof["desc"])

    st.divider()

    # ── GPS ──────────────────────────────────────
    st.markdown("**📡 GPS**")
    gc = st.columns(2)
    gps_hz = gc[0].select_slider(L("rs3.freq"), [1, 2, 5, 10],
                                  value=_prof.get("gps_hz", 1), key="gps_hz")
    gps_noise_on = gc[1].checkbox(L("rs3.gps_noise"), value=True)

    if gps_noise_on:
        gn = st.columns(3)
        sigma_pos = gn[0].slider("Jitter (m)", 0.0, 10.0,
                                  _prof.get("sig_pos", 2.0), 0.5, key="sig_pos")
        n_blackouts = gn[1].number_input(L("rs3.blackouts"), 0, 10, 0, key="n_bo")
        cold_drift = gn[2].slider(L("rs3.cold_drift"), 0.0, 30.0, 0.0, 5.0, key="cold")
    else:
        sigma_pos, n_blackouts, cold_drift = 0.0, 0, 0.0

    st.divider()

    # ── IMU ──────────────────────────────────────
    st.markdown(f"**🔧 {L('rs3.imu_title')}**")
    ic = st.columns(3)
    imu_hz = ic[0].select_slider(L("rs3.freq"), [5, 10],
                                  value=_prof.get("imu_hz", 10), key="imu_hz")
    sigma_acc = ic[1].slider(L("rs3.acc_noise"), 0.0, 0.25,
                              _prof.get("sig_acc", 0.02), 0.005, key="sig_acc")
    gyro_on = ic[2].checkbox(L("rs3.gyro_on"), value=_prof.get("gyro", True))
    sigma_gyro = 0.001 if gyro_on else 0.0

    st.divider()

    # ── Rotation device ──────────────────────────
    st.markdown(f"**🔄 {L('rs3.orientation')}**")
    rc = st.columns(3)
    rot_roll = rc[0].number_input("Roll°", -180., 180.,
                                   _prof.get("roll", 0.0), 1., key="rot_r")
    rot_pitch = rc[1].number_input("Pitch°", -45., 45.,
                                    _prof.get("pitch", 0.0), 1., key="rot_p")
    rot_yaw = rc[2].number_input("Yaw°", -180., 180.,
                                  _prof.get("yaw", 0.0), 1., key="rot_y")

    st.divider()

    # ── Événements ───────────────────────────────
    st.markdown(f"**⚡ {L('rs3.events_title')}**")
    ev1, ev2, ev3 = st.columns(3)
    n_brake = ev1.number_input(L("rs3.brakes"), 0, 20, 0, key="nb")
    n_accel = ev1.number_input(L("rs3.accels"), 0, 20, 0, key="na")
    n_bump = ev2.number_input(L("rs3.bumps"), 0, 20, 0, key="nbu")
    n_pothole = ev2.number_input(L("rs3.potholes"), 0, 20, 0, key="np")
    n_turn = ev3.number_input(L("rs3.turns"), 0, 20, 0, key="nt")
    n_door = ev3.number_input(L("rs3.door"), 0, 20, 0, key="nd")

    with st.expander(L("rs3.carto")):
        osrm_url = st.text_input("OSRM", value=OSRM_URL)

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

st.divider()

def _build_cfg():
    start_dt = datetime.combine(datetime.now(timezone.utc).date(),
                                 datetime.now(timezone.utc).time().replace(second=0, microsecond=0),
                                 tzinfo=timezone.utc)
    stops_yaml = []
    t = start_dt
    for i, s in enumerate(st.session_state.stops):
        entry = {"id": s["id"], "lat": s["lat"], "lon": s["lon"], "service_s": s["service_s"]}
        if i > 0 and s["id"] != "DEPOT":
            entry["tw_start"] = t.isoformat()
            t += timedelta(minutes=15)
            entry["tw_end"] = t.isoformat()
        stops_yaml.append(entry)
    return {
        "outdir": f"data/simulations/WEB-{uuid.uuid4().hex[:8]}",
        "start_time_utc": start_dt.isoformat(),
        "osrm": {"base_url": osrm_url, "profile": "driving"},
        "sim": {"hz": 10},
        "sensors": {"gps_hz": gps_hz, "imu_hz": imu_hz, "gyro_enabled": gyro_on},
        "device_rotation": {"roll_deg": rot_roll, "pitch_deg": rot_pitch, "yaw_deg": rot_yaw},
        "noise_injector": {"sigma_acc": sigma_acc, "sigma_gyro": sigma_gyro},
        "gps_noise": {
            "sigma_pos_m": sigma_pos,
            "sigma_speed_mps": sigma_pos * 0.05,  # proportionnel au jitter position
            "jump_probability": 0.003 if gps_noise_on else 0,
            "jump_max_m": 50,
            "blackout_count": n_blackouts,
            "blackout_min_s": 5,
            "blackout_max_s": 20,
            "cold_start_drift_m": cold_drift,
            "cold_start_decay_s": 30,
        },
        "inject_events": {"n_harsh_brake": n_brake, "n_harsh_accel": n_accel, "n_speed_bump": n_bump,
                          "n_pothole": n_pothole, "n_sharp_turn": n_turn, "n_door_open": n_door},
        "stops": stops_yaml,
    }

can_run = len(st.session_state.stops) >= 2

if st.button(L("rs3.launch"), disabled=not can_run, type="primary", use_container_width=True):
    with st.spinner(L("rs3.running")):
        try:
            import importlib
            rs3_root = str(Path(__file__).resolve().parent.parent)
            if rs3_root not in sys.path:
                sys.path.insert(0, rs3_root)
            mods = [m for m in sys.modules if m.startswith(("engine.", "runner."))]
            for mod in mods:
                try:
                    importlib.reload(sys.modules[mod])
                except Exception:
                    pass

            from engine.context import Context
            from runner.simulate import build_pipeline

            cfg = _build_cfg()
            now = datetime.now(timezone.utc)
            try:
                cfg["outdir"] = now.strftime(cfg["outdir"])
            except Exception:
                pass

            ctx = Context(cfg=cfg)
            pipeline = build_pipeline(cfg)
            result = pipeline.run(ctx)

            st.session_state.sim_ctx = ctx
            st.session_state.sim_outdir = ctx.meta.get("outdir", cfg.get("outdir"))
            st.session_state.sim_df = ctx.df.copy() if ctx.df is not None else None

            if result.ok:
                st.success(f"{L('rs3.success')} {result.msg}")
            else:
                st.error(f"{L('rs3.fail')} {result.msg}")
        except Exception as exc:
            st.error(f"{L('rs3.error')} {exc}")

# ══════════════════════════════════════════════════════════════════════════════
# RÉSULTATS
# ══════════════════════════════════════════════════════════════════════════════

ctx = st.session_state.sim_ctx
sim_df = st.session_state.sim_df
outdir = st.session_state.sim_outdir

if ctx is None:
    st.stop()

st.divider()

# Actions
csv_file = list(Path(outdir).glob("timeline.csv")) if outdir and Path(outdir).exists() else []
act = st.columns([1, 1, 2])
if csv_file:
    with act[0]:
        # D0 Parquet prioritaire, sinon CSV
        d0_file = list(Path(outdir).glob("d0.parquet"))
        if d0_file:
            st.download_button("📥 D0 Parquet", data=d0_file[0].read_bytes(), file_name="d0.parquet",
                               mime="application/octet-stream", key="dl_d0", use_container_width=True)
        else:
            st.download_button("📥 CSV", data=csv_file[0].read_bytes(), file_name="timeline.csv",
                               mime="text/csv", key="dl_csv", use_container_width=True)
        # Copier pour Telemachus (D0 parquet + CSV fallback)
        try:
            if d0_file:
                shutil.copy2(d0_file[0], Path("/opt/shared/traces/rs3_latest_d0.parquet"))
            shutil.copy2(csv_file[0], Path("/opt/shared/traces/rs3_latest.csv"))
        except Exception:
            pass
    with act[1]:
        st.markdown('<a href="https://telemachus.roadsimulator3.fr" target="_blank" style="display:inline-block;'
                    'width:100%;text-align:center;padding:0.4em 0;background:#0066CC;color:white;border-radius:6px;'
                    'text-decoration:none;font-weight:600">📡 Telemachus</a>', unsafe_allow_html=True)

rot_meta = ctx.meta.get("device_rotation_deg")
if rot_meta and any(v != 0 for v in rot_meta.values()):
    act[2].caption(f"Rotation : {rot_meta['roll']}° / {rot_meta['pitch']}° / {rot_meta['yaw']}°")

# Tabs résultats
if sim_df is not None and not sim_df.empty:
    tab_carte, tab_capteurs, tab_severity, tab_qa = st.tabs([L("rs3.tab_map"), L("rs3.tab_sensors"), L("rs3.tab_severity"), L("rs3.tab_quality")])

    with tab_carte:
        hz_obs = float(ctx.meta.get("hz", 10))
        df_map = sim_df.iloc[::max(1, int(hz_obs))]
        valid = df_map["lat"].notna() & df_map["lon"].notna()
        pts = df_map[valid]
        if len(pts) > 0:
            center = [pts["lat"].iloc[len(pts) // 2], pts["lon"].iloc[len(pts) // 2]]
            m_r = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
            coords = list(zip(pts["lat"].values, pts["lon"].values))
            folium.plugins.AntPath(coords, color="blue", weight=3, delay=800).add_to(m_r)
            folium.Marker(coords[0], icon=folium.Icon(color="green", icon="play", prefix="glyphicon")).add_to(m_r)
            folium.Marker(coords[-1], icon=folium.Icon(color="red", icon="stop", prefix="glyphicon")).add_to(m_r)
            st_folium(m_r, width=900, height=450, returned_objects=[])

    with tab_capteurs:
        stride = max(1, len(sim_df) // 5000)
        ds = sim_df.iloc[::stride].copy()
        if "timestamp" in ds.columns:
            ts = pd.to_datetime(ds["timestamp"], utc=True, errors="coerce")
            t0 = ts.dropna().iloc[0] if ts.notna().any() else pd.Timestamp.now(tz="UTC")
            ds["_t"] = (ts - t0).dt.total_seconds()
            xc, xl = "_t", "Temps (s)"
        else:
            ds["_t"] = range(len(ds))
            xc, xl = "_t", "Index"

        # Acc
        acc_cols = [c for c in ["acc_x", "acc_y", "acc_z"] if c in ds.columns]
        if acc_cols:
            fig_a = go.Figure()
            for col, color, label in [("acc_x", "#e74c3c", "Ax"), ("acc_y", "#2ecc71", "Ay"), ("acc_z", "#3498db", "Az")]:
                if col in ds.columns:
                    fig_a.add_trace(go.Scatter(x=ds[xc], y=ds[col], mode="lines", name=label,
                                               line=dict(color=color, width=1)))
            fig_a.update_layout(title=L("rs3.chart_accel"), xaxis_title=xl, yaxis_title="m/s²", height=280,
                                margin=dict(l=50, r=20, t=40, b=30), legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_a, use_container_width=True)

        # Gyro
        gyro_cols = [c for c in ["gyro_x", "gyro_y", "gyro_z"] if c in ds.columns]
        has_gyro = gyro_cols and not all(ds[c].fillna(0).eq(0).all() for c in gyro_cols)
        if has_gyro:
            fig_g = go.Figure()
            for col, color, label in [("gyro_x", "#e67e22", "Gx"), ("gyro_y", "#9b59b6", "Gy"), ("gyro_z", "#1abc9c", "Gz")]:
                if col in ds.columns and not ds[col].isna().all():
                    fig_g.add_trace(go.Scatter(x=ds[xc], y=ds[col], mode="lines", name=label,
                                               line=dict(color=color, width=1)))
            fig_g.update_layout(title=L("rs3.chart_gyro"), xaxis_title=xl, yaxis_title="rad/s", height=250,
                                margin=dict(l=50, r=20, t=40, b=30), legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_g, use_container_width=True)
        else:
            st.caption(L("rs3.gyro_off"))

    with tab_severity:
        from webui.shared.bihistogram import render_bihistogram
        render_bihistogram(sim_df, speed_col="speed", ax_col="acc_x", ay_col="acc_y",
                           hz=float(ctx.meta.get("hz", 10)))

    with tab_qa:
        qa = ctx.artifacts.get("qa_pretty", {})
        if qa:
            if ctx.artifacts.get("qa_realism", {}).get("ok", True):
                st.success(qa.get("status", ""))
            else:
                st.warning(qa.get("status", ""))
            if qa.get("text"):
                st.code(qa["text"], language=None)
        metrics = ctx.artifacts.get("qa_checklist", {}).get("metrics", {})
        if metrics:
            qm = st.columns(4)
            for i, (k, (l, u)) in enumerate({"v_median_mps": (L("rs3.v_median"), "m/s"), "ax_std_mps2": (L("rs3.acc_std"), "m/s²"),
                                              "gz_std_rad_s": (L("rs3.gyro_std"), "rad/s"), "hz_observed": (L("rs3.hz_obs"), "Hz")}.items()):
                if k in metrics:
                    qm[i].metric(l, f"{metrics[k]:.3f} {u}")

# ── Footer ───────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.8em'>"
    "<a href='https://telemachus.roadsimulator3.fr' style='color:#888'>Telemachus</a> · "
    "<a href='https://research.roadsimulator3.fr' style='color:#888'>Recherche</a>"
    "<br>RoadSimulator3 — Telemachus certified output</div>", unsafe_allow_html=True)

"""
RS3 Web UI — simulate.roadsimulator3.fr

Streamlit app: carte Leaflet pour placer des arrêts, config événements,
génération YAML automatique, lancement simulation RS3, téléchargement multi-format.
Affichage résultat : carte avec véhicule animé + graphes acc/gyro synchronisés.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import folium
import folium.plugins
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from streamlit_folium import st_folium

# ── Page config (DOIT être le premier appel Streamlit) ──────────────────────

st.set_page_config(page_title="RS3 Simulator", page_icon="🚛", layout="wide", menu_items={})

from shared.theme import apply_theme, page_header, footer as theme_footer, COLORS
apply_theme()

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_CENTER = [49.38, 1.25]  # Entre Romilly-sur-Andelle et Rouen
DEFAULT_ZOOM = 11
OSRM_BASE_URL = "http://localhost:5000"

# Couleurs événements pour la carte
EVENT_COLORS = {
    "stop": "red",
    "wait": "orange",
    "freinage_final": "darkred",
    "acceleration_initiale": "green",
    "harsh_brake": "#c0392b",
    "harsh_accel": "#27ae60",
    "speed_bump": "#8e44ad",
    "pothole": "#e67e22",
    "curb": "#f39c12",
    "sharp_turn": "#2980b9",
    "door_open": "#1abc9c",
}

EVENT_ICONS = {
    "stop": "pause",
    "wait": "time",
    "freinage_final": "stop",
    "acceleration_initiale": "play",
    "harsh_brake": "warning-sign",
    "harsh_accel": "flash",
    "speed_bump": "road",
    "sharp_turn": "repeat",
}

# ── Session state init ───────────────────────────────────────────────────────

for key, default in [
    ("stops", []),
    ("sim_ctx", None),
    ("sim_outdir", None),
    ("sim_df", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def _next_stop_id() -> str:
    n = len(st.session_state.stops)
    return "DEPOT" if n == 0 else f"C{n}"


# ── Titre ────────────────────────────────────────────────────────────────────

page_header("RoadSimulator3", "Simulateur télématique — Placez des arrêts, configurez, lancez", icon="🚛")

# ── Layout ───────────────────────────────────────────────────────────────────

col_map, col_config = st.columns([3, 2])

# ── Carte (col gauche) ──────────────────────────────────────────────────────

with col_map:
    st.subheader("Carte — cliquez pour ajouter un arrêt")

    m = folium.Map(location=DEFAULT_CENTER, zoom_start=DEFAULT_ZOOM, tiles="OpenStreetMap")

    for i, s in enumerate(st.session_state.stops):
        color = "red" if s["id"] == "DEPOT" else "blue"
        folium.Marker(
            location=[s["lat"], s["lon"]],
            popup=f"{s['id']} ({s['service_s']}s)",
            icon=folium.Icon(color=color, icon="home" if s["id"] == "DEPOT" else "info-sign"),
        ).add_to(m)

    if len(st.session_state.stops) >= 2:
        coords = [[s["lat"], s["lon"]] for s in st.session_state.stops]
        folium.PolyLine(coords, color="green", weight=3, opacity=0.7).add_to(m)

    map_data = st_folium(m, width=700, height=500, returned_objects=["last_clicked"])

    if map_data and map_data.get("last_clicked"):
        clicked = map_data["last_clicked"]
        lat = round(clicked["lat"], 6)
        lng = round(clicked["lng"], 6)

        already = any(
            abs(s["lat"] - lat) < 1e-5 and abs(s["lon"] - lng) < 1e-5
            for s in st.session_state.stops
        )
        if not already:
            stop_id = _next_stop_id()
            st.session_state.stops.append({
                "id": stop_id, "lat": lat, "lon": lng,
                "service_s": 0 if stop_id == "DEPOT" else 30,
            })
            st.rerun()

    subcol1, subcol2, subcol3 = st.columns(3)
    with subcol1:
        if st.button("Ajouter retour DEPOT") and st.session_state.stops:
            depot = st.session_state.stops[0]
            st.session_state.stops.append({"id": "DEPOT", "lat": depot["lat"], "lon": depot["lon"], "service_s": 0})
            st.rerun()
    with subcol2:
        if st.button("Supprimer dernier arrêt") and st.session_state.stops:
            st.session_state.stops.pop()
            st.rerun()
    with subcol3:
        if st.button("Tout effacer"):
            st.session_state.stops = []
            st.session_state.sim_ctx = None
            st.session_state.sim_outdir = None
            st.session_state.sim_df = None
            st.rerun()

# ── Configuration (col droite) ──────────────────────────────────────────────

with col_config:
    st.subheader("Configuration")

    # Arrêts
    st.markdown("**Arrêts**")
    if not st.session_state.stops:
        st.info("Cliquez sur la carte pour placer le DEPOT puis les clients.")
    else:
        for i, s in enumerate(st.session_state.stops):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.text(f"{s['id']} ({s['lat']:.4f}, {s['lon']:.4f})")
            with c2:
                new_svc = st.number_input(
                    "service_s", min_value=0, value=s["service_s"], step=10,
                    key=f"svc_{i}", label_visibility="collapsed",
                )
                st.session_state.stops[i]["service_s"] = new_svc

    st.divider()

    # Paramètres
    st.markdown("**Paramètres**")
    start_date = st.date_input("Date de départ", value=datetime.now(timezone.utc).date())
    start_time = st.time_input("Heure de départ (UTC)",
                                value=datetime.now(timezone.utc).time().replace(second=0, microsecond=0))
    with st.expander("Services carto"):
        osrm_url = st.text_input("OSRM URL", value=OSRM_BASE_URL)
        st.caption("Serveur : localhost:5000. Local : 51.91.125.143:5000")

    st.divider()

    # ── Capteurs ─────────────────────────────────────────────────────────
    st.markdown("**Capteurs**")
    gps_hz = st.select_slider("GPS (Hz)", options=[1, 2, 5, 10], value=1)
    imu_hz = st.select_slider("Accéléromètre (Hz)", options=[5, 10], value=10)
    gyro_enabled = st.checkbox("Gyroscope activé", value=True)
    if gyro_enabled:
        st.caption(f"Gyroscope : {imu_hz} Hz (suit l'accéléromètre)")

    st.divider()

    # ── Événements à injecter (nombre) ───────────────────────────────────
    st.markdown("**Événements à injecter**")
    st.caption("Nombre d'événements synthétiques ajoutés au signal")
    evt_cols = st.columns(3)
    with evt_cols[0]:
        n_harsh_brake = st.number_input("Freinages brusques", 0, 20, 0, 1, key="n_hb")
        n_harsh_accel = st.number_input("Accélérations brusques", 0, 20, 0, 1, key="n_ha")
        n_sharp_turn = st.number_input("Virages serrés", 0, 20, 0, 1, key="n_st")
    with evt_cols[1]:
        n_speed_bump = st.number_input("Dos d'âne", 0, 20, 0, 1, key="n_sb")
        n_pothole = st.number_input("Nids de poule", 0, 20, 0, 1, key="n_ph")
        n_curb = st.number_input("Montées trottoir", 0, 20, 0, 1, key="n_cb")
    with evt_cols[2]:
        n_door_open = st.number_input("Ouvertures porte", 0, 20, 0, 1, key="n_do")

    st.divider()

    # ── Bruit capteur ────────────────────────────────────────────────────
    st.markdown("**Bruit capteur**")
    noise_cols = st.columns(2)
    with noise_cols[0]:
        sigma_acc = st.slider("Acc (m/s2)", 0.0, 0.1, 0.02, 0.005,
                              help="sigma bruit accéléromètre")
    with noise_cols[1]:
        sigma_gyro = st.slider("Gyro (rad/s)", 0.0, 0.01, 0.001, 0.0005,
                               help="sigma bruit gyroscope",
                               disabled=not gyro_enabled)

    st.divider()

    # ── Rotation device ──────────────────────────────────────────────────
    st.markdown("**Orientation du device**")
    st.caption("Simule un capteur monté avec un angle (pour tester le redressement)")
    rot_cols = st.columns(3)
    with rot_cols[0]:
        rot_roll = st.number_input("Roll (deg)", -45.0, 45.0, 0.0, 1.0, help="Axe X")
    with rot_cols[1]:
        rot_pitch = st.number_input("Pitch (deg)", -45.0, 45.0, 0.0, 1.0, help="Axe Y")
    with rot_cols[2]:
        rot_yaw = st.number_input("Yaw (deg)", -180.0, 180.0, 0.0, 1.0, help="Axe Z")

# ── Génération YAML ─────────────────────────────────────────────────────────

st.divider()


def _build_yaml_cfg() -> dict:
    start_dt = datetime.combine(start_date, start_time, tzinfo=timezone.utc)

    stops_yaml = []
    current_time = start_dt
    for i, s in enumerate(st.session_state.stops):
        entry = {"id": s["id"], "lat": s["lat"], "lon": s["lon"], "service_s": s["service_s"]}
        if i > 0 and s["id"] != "DEPOT":
            entry["tw_start"] = current_time.isoformat()
            current_time += timedelta(minutes=15)
            entry["tw_end"] = current_time.isoformat()
        stops_yaml.append(entry)

    cfg = {
        "outdir": f"data/simulations/WEB-{uuid.uuid4().hex[:8]}",
        "start_time_utc": start_dt.isoformat(),
        "osrm": {"base_url": osrm_url, "profile": "driving"},
        "sim": {"hz": 10},
        "sensors": {
            "gps_hz": gps_hz,
            "imu_hz": imu_hz,
            "gyro_enabled": gyro_enabled,
        },
        "stop_smoother": {"v_in": 0.25, "t_in": 2.0, "v_out": 0.6, "t_out": 2.5, "lock_pos": True},
        "noise_injector": {
            "sigma_acc": sigma_acc,
            "sigma_gyro": sigma_gyro if gyro_enabled else 0.0,
        },
        "inject_events": {
            "n_harsh_brake": n_harsh_brake,
            "n_harsh_accel": n_harsh_accel,
            "n_speed_bump": n_speed_bump,
            "n_pothole": n_pothole,
            "n_curb": n_curb,
            "n_sharp_turn": n_sharp_turn,
            "n_door_open": n_door_open,
        },
        "device_rotation": {
            "roll_deg": rot_roll,
            "pitch_deg": rot_pitch,
            "yaw_deg": rot_yaw,
        },
        "stops": stops_yaml,
    }
    return cfg


if st.session_state.stops:
    yaml_cfg = _build_yaml_cfg()
    yaml_str = yaml.dump(yaml_cfg, default_flow_style=False, allow_unicode=True, sort_keys=False)
    with st.expander("YAML généré", expanded=False):
        st.code(yaml_str, language="yaml")
else:
    yaml_str = ""
    st.info("Ajoutez des arrêts pour voir le YAML.")

# ── Lancer la simulation ────────────────────────────────────────────────────

st.divider()
st.subheader("Simulation")

run_mode = st.radio("Mode", ["Local (import direct)", "API (FastAPI)"], horizontal=True)
can_run = len(st.session_state.stops) >= 2

if st.button("Lancer la simulation", disabled=not can_run, type="primary"):
    if run_mode == "Local (import direct)":
        progress_bar = st.progress(0, text="Initialisation...")
        try:
            import sys
            import importlib
            rs3_root = str(Path(__file__).resolve().parent.parent)
            if rs3_root not in sys.path:
                sys.path.insert(0, rs3_root)

            # Force reload de TOUS les modules RS3 pour éviter le cache Streamlit
            mods_to_reload = [m for m in sys.modules if m.startswith(("core2.", "runner."))]
            for mod_name in mods_to_reload:
                try:
                    importlib.reload(sys.modules[mod_name])
                except Exception:
                    pass

            from core2.context import Context
            from runner.run_simulation2 import build_pipeline
            importlib.reload(sys.modules["runner.run_simulation2"])

            cfg = _build_yaml_cfg()

            # Debug : afficher la config rotation et sensors
            rot_cfg = cfg.get("device_rotation", {})
            sens_cfg = cfg.get("sensors", {})
            st.info(f"Config: rotation={rot_cfg}, sensors={sens_cfg}")

            now = datetime.now(timezone.utc)
            if isinstance(cfg.get("outdir"), str):
                try:
                    cfg["outdir"] = now.strftime(cfg["outdir"])
                except Exception:
                    pass

            progress_bar.progress(10, text="Construction du pipeline...")
            ctx = Context(cfg=cfg)
            pipeline = build_pipeline(cfg)

            progress_bar.progress(20, text="Simulation en cours...")
            result = pipeline.run(ctx)

            progress_bar.progress(100, text="Terminé !")

            st.session_state.sim_ctx = ctx
            st.session_state.sim_outdir = ctx.meta.get("outdir", cfg.get("outdir"))
            if ctx.df is not None and not ctx.df.empty:
                st.session_state.sim_df = ctx.df.copy()

            if result.ok:
                st.success(f"Simulation terminée — {result.msg}")
            else:
                st.error(f"Simulation échouée — {result.msg}")

        except Exception as exc:
            progress_bar.empty()
            st.error(f"Erreur: {exc}")
            import traceback
            st.code(traceback.format_exc())

    else:
        with st.spinner("Appel API en cours..."):
            try:
                import httpx
                resp = httpx.post("http://localhost:8100/simulate", json=_build_yaml_cfg(), timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.sim_outdir = data.get("outdir")
                    st.success(f"OK — {data.get('message', 'done')}")
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except Exception as exc:
                st.error(f"Erreur API: {exc}")

# ── Analyser dans Telemachus (1 clic) ─────────────────────────────────────

outdir = st.session_state.sim_outdir
if outdir and Path(outdir).exists():
    csv_candidates = list(Path(outdir).glob("timeline.csv"))
    if csv_candidates:
        # Copier le CSV dans le dossier partagé pour Telemachus
        import shutil
        shared_dir = Path("/opt/shared/traces")
        if not shared_dir.exists():
            shared_dir = Path(outdir)  # fallback local
        shared_csv = shared_dir / f"rs3_latest.csv"
        try:
            shutil.copy2(csv_candidates[0], shared_csv)
        except Exception:
            shared_csv = csv_candidates[0]

        tel_url = "https://telemachus.roadsimulator3.fr"
        st.markdown(
            f'<a href="{tel_url}" target="_blank" style="display:inline-block;'
            f'padding:0.5em 1.5em;background:#0066CC;color:white;border-radius:8px;'
            f'text-decoration:none;font-weight:600">📡 Ouvrir dans Telemachus →</a>'
            f'<br><small style="color:#888">Fichier prêt : uploadez <code>{shared_csv.name}</code> '
            f'dans Telemachus ({len(pd.read_csv(csv_candidates[0], nrows=1).columns)} colonnes)</small>',
            unsafe_allow_html=True,
        )

        # Download direct du CSV
        st.download_button(
            "📥 Télécharger timeline.csv",
            data=csv_candidates[0].read_bytes(),
            file_name="timeline.csv",
            mime="text/csv",
            key="dl_timeline",
        )

# ── Rapport qualité ─────────────────────────────────────────────────────────

ctx = st.session_state.sim_ctx
if ctx is not None:
    st.divider()
    st.subheader("Rapport qualité")

    qa_pretty = ctx.artifacts.get("qa_pretty", {})
    qa_realism = ctx.artifacts.get("qa_realism", {})

    if qa_pretty:
        status = qa_pretty.get("status", "")
        text = qa_pretty.get("text", "")
        is_ok = qa_realism.get("ok", True) if qa_realism else True

        if is_ok:
            st.success(status)
        else:
            st.warning(status)

        if text:
            st.code(text, language=None)

    # Métriques clés
    qa_checklist = ctx.artifacts.get("qa_checklist", {})
    metrics = qa_checklist.get("metrics", {})
    if metrics:
        mcols = st.columns(4)
        metric_labels = {
            "v_median_mps": ("Vitesse médiane", "m/s"),
            "ax_std_mps2": ("Std acc_x", "m/s2"),
            "gz_std_rad_s": ("Std gyro_z", "rad/s"),
            "hz_observed": ("Hz observé", "Hz"),
        }
        for i, (key, (label, unit)) in enumerate(metric_labels.items()):
            if key in metrics:
                val = metrics[key]
                mcols[i % 4].metric(label, f"{val:.3f} {unit}")

    # Rotation appliquée
    rot_meta = ctx.meta.get("device_rotation_deg")
    if rot_meta and any(v != 0 for v in rot_meta.values()):
        st.info(f"Rotation device appliquée : roll={rot_meta['roll']}° pitch={rot_meta['pitch']}° yaw={rot_meta['yaw']}°")

# ══════════════════════════════════════════════════════════════════════════════
# ── RÉSULTATS VISUELS ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

sim_df = st.session_state.sim_df
if sim_df is not None and not sim_df.empty:
    st.divider()
    st.subheader("Résultats")

    # ── Comptage correct des événements ──────────────────────────────────
    def _count_events(df: pd.DataFrame) -> dict[str, int]:
        """Compte les blocs d'événements distincts (pas les points)."""
        if "event" not in df.columns:
            return {}
        ev = df["event"].astype(str).fillna("").str.strip()
        # Détecter les transitions : un bloc = suite contiguë du même type
        shifted = ev.shift(fill_value="")
        is_new_block = (ev != "") & (ev != shifted)
        counts: dict[str, int] = {}
        for idx in is_new_block[is_new_block].index:
            t = ev.at[idx]
            counts[t] = counts.get(t, 0) + 1
        return counts

    event_counts = _count_events(sim_df)

    if event_counts:
        st.markdown("**Événements détectés :**")
        ev_display_cols = st.columns(min(len(event_counts), 4))
        for i, (evt_name, cnt) in enumerate(sorted(event_counts.items())):
            color = EVENT_COLORS.get(evt_name, "#95a5a6")
            ev_display_cols[i % len(ev_display_cols)].markdown(
                f"<span style='color:{color}; font-weight:bold'>{evt_name}</span> : {cnt}",
                unsafe_allow_html=True,
            )

    tab_map, tab_acc, tab_gyro = st.tabs(["Carte du trajet", "Accéléromètre", "Gyroscope"])

    # ── Tab carte : véhicule animé ───────────────────────────────────────
    with tab_map:
        lat_col = "lat" if "lat" in sim_df.columns else None
        lon_col = "lon" if "lon" in sim_df.columns else None

        if lat_col and lon_col:
            # Sous-échantillonner (1 Hz max pour la carte)
            hz_obs = float(ctx.meta.get("hz", 10)) if ctx else 10
            stride = max(1, int(hz_obs))
            df_map = sim_df.iloc[::stride].copy()

            valid = df_map[lat_col].notna() & df_map[lon_col].notna()
            df_pts = df_map[valid]

            if len(df_pts) > 0:
                center = [df_pts[lat_col].iloc[len(df_pts)//2],
                          df_pts[lon_col].iloc[len(df_pts)//2]]
                m_result = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

                coords = list(zip(df_pts[lat_col].values, df_pts[lon_col].values))

                # Route animée (flèches directionnelles = véhicule qui avance)
                folium.plugins.AntPath(
                    locations=coords,
                    color="blue",
                    weight=3,
                    opacity=0.7,
                    delay=800,
                    dash_array=[20, 30],
                    pulse_color="#3498db",
                ).add_to(m_result)

                # Marqueur véhicule au départ
                folium.Marker(
                    location=coords[0],
                    popup="Départ",
                    icon=folium.Icon(color="green", icon="play", prefix="glyphicon"),
                ).add_to(m_result)

                # Marqueur véhicule à l'arrivée
                folium.Marker(
                    location=coords[-1],
                    popup="Arrivée",
                    icon=folium.Icon(color="red", icon="stop", prefix="glyphicon"),
                ).add_to(m_result)

                # Marqueurs événements (sur le df complet, pas sous-échantillonné)
                if "event" in sim_df.columns:
                    ev = sim_df["event"].astype(str).fillna("").str.strip()
                    shifted = ev.shift(fill_value="")
                    ev_starts = sim_df[(ev != "") & (ev != shifted)]

                    for _, row in ev_starts.iterrows():
                        ev_type = str(row["event"]).strip()
                        rlat, rlon = row.get("lat"), row.get("lon")
                        if pd.isna(rlat) or pd.isna(rlon):
                            continue
                        color = EVENT_COLORS.get(ev_type, "gray")
                        icon_name = EVENT_ICONS.get(ev_type, "info-sign")
                        folium.CircleMarker(
                            location=[float(rlat), float(rlon)],
                            radius=7,
                            color=color,
                            fill=True,
                            fill_opacity=0.9,
                            popup=ev_type,
                            tooltip=ev_type,
                        ).add_to(m_result)

                st_folium(m_result, width=900, height=500, returned_objects=[])
            else:
                st.warning("Pas de coordonnées GPS valides.")
        else:
            st.warning("Colonnes lat/lon absentes.")

    # ── Tab accéléromètre ────────────────────────────────────────────────
    with tab_acc:
        acc_cols_present = [c for c in ["acc_x", "acc_y", "acc_z"] if c in sim_df.columns]
        if acc_cols_present:
            import plotly.graph_objects as go

            max_pts = 5000
            stride_plot = max(1, len(sim_df) // max_pts)
            df_plot = sim_df.iloc[::stride_plot].copy()

            if "timestamp" in df_plot.columns:
                ts = pd.to_datetime(df_plot["timestamp"], utc=True, errors="coerce")
                t0 = ts.iloc[0]
                df_plot["_t_s"] = (ts - t0).dt.total_seconds()
                x_col, x_label = "_t_s", "Temps (s)"
            else:
                df_plot["_idx"] = range(len(df_plot))
                x_col, x_label = "_idx", "Index"

            fig_acc = go.Figure()
            styles = {
                "acc_x": ("#e74c3c", "Ax (longitudinal)"),
                "acc_y": ("#2ecc71", "Ay (latéral)"),
                "acc_z": ("#3498db", "Az (vertical)"),
            }
            for col in acc_cols_present:
                vals = pd.to_numeric(df_plot[col], errors="coerce")
                color, label = styles[col]
                fig_acc.add_trace(go.Scatter(
                    x=df_plot[x_col], y=vals,
                    mode="lines", name=label,
                    line=dict(color=color, width=1),
                ))

            # Marqueurs événements injectés
            if "event" in sim_df.columns:
                ev_full = sim_df.iloc[::stride_plot].copy()
                if "timestamp" in ev_full.columns:
                    ts_ev = pd.to_datetime(ev_full["timestamp"], utc=True, errors="coerce")
                    ev_full["_t_s"] = (ts_ev - t0).dt.total_seconds()
                ev_mask = ev_full["event"].astype(str).str.strip().ne("")
                ev_pts = ev_full[ev_mask]
                if len(ev_pts) > 0 and x_col in ev_pts.columns and "acc_x" in ev_pts.columns:
                    # Colorer par type
                    for evt_type in ev_pts["event"].unique():
                        sub = ev_pts[ev_pts["event"] == evt_type]
                        color = EVENT_COLORS.get(str(evt_type).strip(), "black")
                        fig_acc.add_trace(go.Scatter(
                            x=sub[x_col],
                            y=pd.to_numeric(sub["acc_x"], errors="coerce"),
                            mode="markers", name=str(evt_type),
                            marker=dict(color=color, size=5, symbol="diamond"),
                            showlegend=True,
                        ))

            fig_acc.update_layout(
                title="Accéléromètre",
                xaxis_title=x_label, yaxis_title="Accélération (m/s2)",
                height=350, margin=dict(l=50, r=20, t=40, b=40),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_acc, use_container_width=True)
        else:
            st.info("Pas de données accéléromètre.")

    # ── Tab gyroscope ────────────────────────────────────────────────────
    with tab_gyro:
        gyro_cols_present = [c for c in ["gyro_x", "gyro_y", "gyro_z"] if c in sim_df.columns]
        has_gyro = gyro_cols_present and not all(sim_df[c].isna().all() for c in gyro_cols_present)

        if has_gyro:
            import plotly.graph_objects as go

            max_pts = 5000
            stride_plot = max(1, len(sim_df) // max_pts)
            df_g = sim_df.iloc[::stride_plot].copy()

            if "timestamp" in df_g.columns:
                ts = pd.to_datetime(df_g["timestamp"], utc=True, errors="coerce")
                t0 = ts.iloc[0]
                df_g["_t_s"] = (ts - t0).dt.total_seconds()
                x_col_g, x_label_g = "_t_s", "Temps (s)"
            else:
                df_g["_idx"] = range(len(df_g))
                x_col_g, x_label_g = "_idx", "Index"

            fig_gyro = go.Figure()
            styles_g = {
                "gyro_x": ("#e67e22", "Gx (roll)"),
                "gyro_y": ("#9b59b6", "Gy (pitch)"),
                "gyro_z": ("#1abc9c", "Gz (yaw)"),
            }
            for col in gyro_cols_present:
                vals = pd.to_numeric(df_g[col], errors="coerce")
                if vals.isna().all():
                    continue
                color, label = styles_g[col]
                fig_gyro.add_trace(go.Scatter(
                    x=df_g[x_col_g], y=vals,
                    mode="lines", name=label,
                    line=dict(color=color, width=1),
                ))

            fig_gyro.update_layout(
                title="Gyroscope",
                xaxis_title=x_label_g, yaxis_title="Vitesse angulaire (rad/s)",
                height=350, margin=dict(l=50, r=20, t=40, b=40),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_gyro, use_container_width=True)
        else:
            st.info("Gyroscope désactivé ou pas de données gyro.")

# ── Téléchargement ──────────────────────────────────────────────────────────

outdir = st.session_state.sim_outdir
if outdir and Path(outdir).exists():
    st.divider()
    st.subheader("Téléchargement")

    out_path = Path(outdir)
    parquet_files = list(out_path.glob("*.parquet"))
    csv_files = list(out_path.glob("*.csv"))
    json_files = list(out_path.glob("*.json"))

    st.markdown(f"**Dossier:** `{outdir}`")

    dl_cols = st.columns(4)

    with dl_cols[0]:
        for pf in parquet_files:
            st.download_button(f"Parquet: {pf.name}", data=pf.read_bytes(),
                               file_name=pf.name, mime="application/octet-stream", key=f"dl_pq_{pf.name}")

    with dl_cols[1]:
        if csv_files:
            for cf in csv_files:
                st.download_button(f"CSV: {cf.name}", data=cf.read_bytes(),
                                   file_name=cf.name, mime="text/csv", key=f"dl_csv_{cf.name}")
        elif parquet_files:
            for pf in parquet_files:
                df_conv = pd.read_parquet(pf)
                buf = io.StringIO()
                df_conv.to_csv(buf, index=False)
                st.download_button(f"CSV: {pf.stem}.csv", data=buf.getvalue().encode(),
                                   file_name=f"{pf.stem}.csv", mime="text/csv", key=f"dl_c_{pf.name}")

    with dl_cols[2]:
        if json_files:
            for jf in json_files:
                st.download_button(f"JSON: {jf.name}", data=jf.read_bytes(),
                                   file_name=jf.name, mime="application/json", key=f"dl_json_{jf.name}")
        elif parquet_files:
            for pf in parquet_files:
                df_conv = pd.read_parquet(pf)
                jbuf = df_conv.to_json(orient="records", date_format="iso")
                st.download_button(f"JSON: {pf.stem}.json", data=jbuf.encode(),
                                   file_name=f"{pf.stem}.json", mime="application/json", key=f"dl_j_{pf.name}")

    with dl_cols[3]:
        if yaml_str:
            st.download_button("YAML config", data=yaml_str.encode(),
                               file_name="simulator_web.yaml", mime="text/yaml", key="dl_yaml")

# ── Footer ──────────────────────────────────────────────────────────────────

st.markdown("[📡 Analyser dans Telemachus](https://telemachus.roadsimulator3.fr) · [📄 Recherche](https://research.roadsimulator3.fr)")

theme_footer("RoadSimulator3 — Telemachus certified output")

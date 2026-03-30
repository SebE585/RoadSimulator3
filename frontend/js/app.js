/**
 * RS3 Simulator — Main application
 * Orchestre map, config, simulation et resultats
 */

import { simulate, getTrace, getSensors, getMeta, getQA, getSeverity, downloadUrl } from "./api.js";
import { initInputMap, addStopMarker, updateRoute, clearInputMap, refreshMarkers, initResultMap, showTrace } from "./map.js";
import { plotAccelerometer, plotGyroscope, plotSpeed, plotBiHistogram, renderQA } from "./charts.js";
import { detectLocale, setLocale, applyTranslations, t } from "./i18n.js";

// ── State ──────────────────────────────────────────────────

const state = {
    stops: [],
    jobId: null,
};

// ── Init ───────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    // i18n : détection locale + sélecteur
    const detectedLocale = detectLocale();
    const langSelector = document.getElementById("lang-selector");
    if (langSelector) {
        langSelector.value = detectedLocale;
        langSelector.addEventListener("change", () => {
            setLocale(langSelector.value);
            applyTranslations();
            updateUI();
        });
    }
    applyTranslations();

    // Carte de saisie
    initInputMap("map", onMapClick);

    // Boutons
    document.getElementById("btn-return-depot").addEventListener("click", returnToDepot);
    document.getElementById("btn-remove-last").addEventListener("click", removeLast);
    document.getElementById("btn-clear").addEventListener("click", clearAll);
    document.getElementById("btn-simulate").addEventListener("click", runSimulation);

    // Tabs
    document.querySelectorAll(".tab").forEach((tab) => {
        tab.addEventListener("click", () => switchTab(tab.dataset.tab));
    });

    updateUI();
});

// ── Map click ──────────────────────────────────────────────

function onMapClick(lat, lng) {
    const n = state.stops.length;
    const id = n === 0 ? "DEPOT" : `C${n}`;
    const stop = { id, lat, lon: lng, service_s: id === "DEPOT" ? 0 : 30 };

    state.stops.push(stop);
    addStopMarker(stop, n);
    updateRoute(state.stops);
    updateUI();
}

// ── Stop management ────────────────────────────────────────

function returnToDepot() {
    if (state.stops.length === 0) return;
    const depot = state.stops[0];
    state.stops.push({ id: "DEPOT", lat: depot.lat, lon: depot.lon, service_s: 0 });
    refreshMarkers(state.stops);
    updateUI();
}

function removeLast() {
    if (state.stops.length === 0) return;
    state.stops.pop();
    refreshMarkers(state.stops);
    updateUI();
}

function clearAll() {
    state.stops = [];
    state.jobId = null;
    clearInputMap();
    updateUI();
    hideResults();
}

// ── UI update ──────────────────────────────────────────────

function updateUI() {
    const list = document.getElementById("stops-list");
    const btn = document.getElementById("btn-simulate");

    // Stops list
    if (state.stops.length === 0) {
        list.innerHTML = '<div class="map-hint">Cliquez sur la carte pour placer le DEPOT puis les clients.</div>';
    } else {
        list.innerHTML = state.stops.map((s, i) => `
            <div class="stop-item">
                <span class="badge ${s.id === 'DEPOT' ? 'depot' : ''}">${s.id}</span>
                <span>${s.lat.toFixed(4)}, ${s.lon.toFixed(4)}</span>
                <input type="number" value="${s.service_s}" min="0" step="10"
                       onchange="window._updateService(${i}, this.value)" />
                <span style="color:#888;font-size:0.8em">s</span>
            </div>
        `).join("");
    }

    // Simulate button
    btn.disabled = state.stops.length < 2;
}

// Expose pour les inputs inline
window._updateService = (index, value) => {
    state.stops[index].service_s = parseInt(value) || 0;
};

// Profils de conduite
const DRIVE_PROFILES = {
    smooth: {
        sigma_acc: 0.01, n_brake: 0, n_accel: 0, n_bump: 0,
        n_pothole: 0, n_turn: 0, n_door: 0, label: "Eco-conduite",
    },
    normal: {
        sigma_acc: 0.02, n_brake: 2, n_accel: 1, n_bump: 1,
        n_pothole: 0, n_turn: 2, n_door: 1, label: "Normal",
    },
    aggressive: {
        sigma_acc: 0.05, n_brake: 5, n_accel: 5, n_bump: 2,
        n_pothole: 1, n_turn: 4, n_door: 2, label: "Agressif",
    },
};

window._applyProfile = (profile) => {
    const p = DRIVE_PROFILES[profile];
    if (!p) return; // custom = ne rien changer

    const set = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.value = val;
    };

    set("cfg-sigma-acc", p.sigma_acc);
    set("cfg-n-brake", p.n_brake);
    set("cfg-n-accel", p.n_accel);
    set("cfg-n-bump", p.n_bump);
    set("cfg-n-pothole", p.n_pothole);
    set("cfg-n-turn", p.n_turn);
    set("cfg-n-door", p.n_door);
};

// ── Config builder ─────────────────────────────────────────

function buildConfig() {
    const val = (id) => {
        const el = document.getElementById(id);
        if (!el) return null;
        if (el.type === "checkbox") return el.checked;
        if (el.type === "number" || el.type === "range") return parseFloat(el.value);
        return el.value;
    };

    const now = new Date().toISOString();
    const stops = state.stops.map((s, i) => {
        const entry = { id: s.id, lat: s.lat, lon: s.lon, service_s: s.service_s };
        return entry;
    });

    return {
        outdir: `data/simulations/WEB-${Date.now().toString(36)}`,
        start_time_utc: now,
        osrm: { base_url: val("cfg-osrm") || "http://localhost:5000", profile: "driving" },
        sim: { hz: 10 },
        sensors: {
            gps_hz: parseInt(val("cfg-gps-hz")) || 1,
            imu_hz: parseInt(val("cfg-imu-hz")) || 10,
            gyro_enabled: val("cfg-gyro"),
        },
        device_rotation: {
            roll_deg: val("cfg-roll") || 0,
            pitch_deg: val("cfg-pitch") || 0,
            yaw_deg: val("cfg-yaw") || 0,
        },
        noise_injector: {
            sigma_acc: val("cfg-sigma-acc") || 0.02,
            sigma_gyro: val("cfg-gyro") ? 0.001 : 0,
        },
        gps_noise: {
            sigma_pos_m: val("cfg-gps-noise") ? (val("cfg-sigma-pos") || 2.0) : 0,
            sigma_speed_mps: val("cfg-gps-noise") ? (val("cfg-sigma-pos") || 2.0) * 0.05 : 0,
            jump_probability: val("cfg-gps-noise") ? 0.003 : 0,
            jump_max_m: 50,
            blackout_count: parseInt(val("cfg-blackouts")) || 0,
            blackout_min_s: 5,
            blackout_max_s: 20,
            cold_start_drift_m: val("cfg-cold-drift") || 0,
            cold_start_decay_s: 30,
        },
        inject_events: {
            n_harsh_brake: parseInt(val("cfg-n-brake")) || 0,
            n_harsh_accel: parseInt(val("cfg-n-accel")) || 0,
            n_speed_bump: parseInt(val("cfg-n-bump")) || 0,
            n_pothole: parseInt(val("cfg-n-pothole")) || 0,
            n_sharp_turn: parseInt(val("cfg-n-turn")) || 0,
            n_door_open: parseInt(val("cfg-n-door")) || 0,
        },
        stops,
    };
}

// ── Simulation ─────────────────────────────────────────────

async function runSimulation() {
    const statusEl = document.getElementById("status");
    const resultsEl = document.getElementById("results");

    statusEl.className = "status loading";
    statusEl.textContent = t("sim.loading");
    statusEl.style.display = "block";
    resultsEl.style.display = "none";

    try {
        const config = buildConfig();
        const result = await simulate(config);

        state.jobId = result.job_id;

        if (result.status === "done") {
            statusEl.className = "status success";
            statusEl.textContent = `${t("sim.done")} — ${result.message}`;
            // Petit delai pour laisser les fichiers s'ecrire sur disque
            await new Promise((r) => setTimeout(r, 500));
            await loadResults(result.job_id);
        } else {
            statusEl.className = "status error";
            statusEl.textContent = `Echec — ${result.message}`;
        }
    } catch (err) {
        statusEl.className = "status error";
        statusEl.textContent = `${t("sim.error")}: ${err.message}`;
    }
}

// ── Results ────────────────────────────────────────────────

async function loadResults(jobId) {
    const resultsEl = document.getElementById("results");
    resultsEl.style.display = "block";

    // Init result map (lazy)
    setTimeout(() => {
        const container = document.getElementById("result-map");
        if (container && !container._leaflet_id) {
            initResultMap("result-map");
        }
    }, 100);

    // Charger tout en parallele (avec fallback sur erreur)
    const empty_sensors = { time_s: [], n_points: 0 };
    const [traceData, sensorData, metaData, qaData, severityData] = await Promise.all([
        getTrace(jobId, 1).catch((e) => { console.warn("trace:", e); return { coords: [] }; }),
        getSensors(jobId, 3000).catch((e) => { console.warn("sensors:", e); return empty_sensors; }),
        getMeta(jobId).catch((e) => { console.warn("meta:", e); return {}; }),
        getQA(jobId).catch((e) => { console.warn("qa:", e); return {}; }),
        getSeverity(jobId).catch((e) => { console.warn("severity:", e); return { gx_mg: [], gy_mg: [] }; }),
    ]);

    // Summary bar
    renderSummary(metaData, traceData, sensorData);

    // Carte
    setTimeout(() => showTrace(traceData), 200);

    // Graphiques capteurs
    plotAccelerometer("chart-acc", sensorData);
    plotGyroscope("chart-gyro", sensorData);
    plotSpeed("chart-speed", sensorData);

    // Severity bi-histogram
    plotBiHistogram("chart-severity", severityData);

    // QA (avec meta pour contextualiser la rotation)
    renderQA("qa-content", qaData, metaData);

    // Download
    const dlBtn = document.getElementById("btn-download");
    if (dlBtn) {
        dlBtn.href = downloadUrl(jobId, "parquet");
        dlBtn.style.display = "inline-block";
    }

    // Switch to carte tab
    switchTab("tab-carte");
}

function hideResults() {
    document.getElementById("results").style.display = "none";
    document.getElementById("status").style.display = "none";
}

// ── Tabs ───────────────────────────────────────────────────

function renderSummary(meta, trace, sensors) {
    const el = document.getElementById("result-summary");
    if (!el) return;

    const rot = meta.device_rotation_deg || {};
    const hasRot = rot.roll || rot.pitch || rot.yaw;
    const duration = sensors.time_s ? sensors.time_s[sensors.time_s.length - 1] : 0;

    el.innerHTML = `
        <div class="summary-grid">
            <div class="summary-item"><span class="summary-value">${trace.n_points || 0}</span><span class="summary-label">${t("summary.gps_points")}</span></div>
            <div class="summary-item"><span class="summary-value">${(duration / 60).toFixed(0)} min</span><span class="summary-label">${t("summary.duration")}</span></div>
            <div class="summary-item"><span class="summary-value">${meta.gps_hz || "?"} / ${meta.imu_hz || "?"} Hz</span><span class="summary-label">${t("summary.freq")}</span></div>
            <div class="summary-item"><span class="summary-value">${meta.gyro_enabled ? "ON" : "OFF"}</span><span class="summary-label">${t("summary.gyro")}</span></div>
            ${hasRot ? `<div class="summary-item"><span class="summary-value">${rot.roll}\u00B0/${rot.pitch}\u00B0/${rot.yaw}\u00B0</span><span class="summary-label">${t("summary.rotation")}</span></div>` : ""}
        </div>
    `;
    el.style.display = "block";
}

function switchTab(tabId) {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === tabId));
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.toggle("active", c.id === tabId));

    // Resize maps/charts after tab switch
    setTimeout(() => {
        if (tabId === "tab-carte") {
            const rm = document.getElementById("result-map");
            if (rm && rm._leaflet_id) {
                const map = Object.values(rm).find((v) => v && v.invalidateSize);
                if (map) map.invalidateSize();
            }
        }
        window.dispatchEvent(new Event("resize"));
    }, 100);
}

/**
 * RS3 Simulator — Charts module
 * Plotly.js pour les graphiques capteurs + bi-histogramme severite
 */

/** Trace les courbes accelerometre */
export function plotAccelerometer(containerId, sensorData) {
    const time = sensorData.time_s;
    const traces = [];

    const axisConfig = [
        { col: "acc_x", name: "Ax (longitudinal)", color: "#e74c3c" },
        { col: "acc_y", name: "Ay (lateral)", color: "#27ae60" },
        { col: "acc_z", name: "Az (vertical)", color: "#3498db" },
    ];

    for (const cfg of axisConfig) {
        if (sensorData[cfg.col]) {
            traces.push({
                x: time,
                y: sensorData[cfg.col],
                mode: "lines",
                name: cfg.name,
                line: { color: cfg.color, width: 1 },
            });
        }
    }

    if (traces.length === 0) {
        document.getElementById(containerId).innerHTML =
            '<p style="color:#888;text-align:center;padding:40px">Pas de donnees accelerometre</p>';
        return;
    }

    Plotly.newPlot(containerId, traces, {
        title: { text: "Accelerometre", font: { size: 14 } },
        xaxis: { title: "Temps (s)" },
        yaxis: { title: "m/s\u00B2" },
        height: 300,
        margin: { l: 50, r: 20, t: 40, b: 40 },
        legend: { orientation: "h", y: -0.25 },
    }, { responsive: true });
}

/** Trace les courbes gyroscope */
export function plotGyroscope(containerId, sensorData) {
    const time = sensorData.time_s;
    const traces = [];

    const axisConfig = [
        { col: "gyro_x", name: "Gx", color: "#e67e22" },
        { col: "gyro_y", name: "Gy", color: "#9b59b6" },
        { col: "gyro_z", name: "Gz", color: "#1abc9c" },
    ];

    for (const cfg of axisConfig) {
        if (sensorData[cfg.col]) {
            const allZero = sensorData[cfg.col].every((v) => v === 0);
            if (!allZero) {
                traces.push({
                    x: time,
                    y: sensorData[cfg.col],
                    mode: "lines",
                    name: cfg.name,
                    line: { color: cfg.color, width: 1 },
                });
            }
        }
    }

    if (traces.length === 0) {
        document.getElementById(containerId).innerHTML =
            '<p style="color:#888;text-align:center;padding:40px">Gyroscope desactive</p>';
        return;
    }

    Plotly.newPlot(containerId, traces, {
        title: { text: "Gyroscope", font: { size: 14 } },
        xaxis: { title: "Temps (s)" },
        yaxis: { title: "rad/s" },
        height: 280,
        margin: { l: 50, r: 20, t: 40, b: 40 },
        legend: { orientation: "h", y: -0.25 },
    }, { responsive: true });
}

/** Trace la courbe vitesse */
export function plotSpeed(containerId, sensorData) {
    if (!sensorData.speed) {
        document.getElementById(containerId).innerHTML =
            '<p style="color:#888;text-align:center;padding:40px">Pas de donnees vitesse</p>';
        return;
    }

    const time = sensorData.time_s;
    const speedKmh = sensorData.speed.map((v) => v * 3.6);

    Plotly.newPlot(containerId, [{
        x: time,
        y: speedKmh,
        mode: "lines",
        name: "Vitesse",
        line: { color: "#0066CC", width: 1.5 },
        fill: "tozeroy",
        fillcolor: "rgba(0,102,204,0.1)",
    }], {
        title: { text: "Vitesse", font: { size: 14 } },
        xaxis: { title: "Temps (s)" },
        yaxis: { title: "km/h" },
        height: 280,
        margin: { l: 50, r: 20, t: 40, b: 40 },
    }, { responsive: true });
}

/** Bi-histogramme de severite (acceleration laterale vs longitudinale) */
export function plotBiHistogram(containerId, severityData) {
    const gx = severityData.gx_mg;
    const gy = severityData.gy_mg;
    const rotation = severityData.rotation || {};

    if (!gx || !gy || gx.length === 0) {
        document.getElementById(containerId).innerHTML =
            '<p style="color:#888;text-align:center;padding:40px">Pas de donnees severity</p>';
        return;
    }

    const hasRotation = rotation.roll !== 0 || rotation.pitch !== 0 || rotation.yaw !== 0;
    const title = hasRotation
        ? `Bi-histogramme (rotation: ${rotation.roll}\u00B0/${rotation.pitch}\u00B0/${rotation.yaw}\u00B0)`
        : "Bi-histogramme (sans rotation)";

    Plotly.newPlot(containerId, [{
        x: gx,
        y: gy,
        type: "histogram2d",
        colorscale: "Jet",
        zsmooth: "best",
        nbinsx: 100,
        nbinsy: 100,
        colorbar: { title: { text: "log10(n)", side: "right" }, thickness: 15 },
        zauto: false,
        zmin: 0,
        zmax: Math.log10(gx.length) + 0.5,
        histfunc: "count",
    }], {
        title: { text: title, font: { size: 14 } },
        xaxis: {
            title: "Lateral Gy (mG)",
            range: [-1000, 1000],
        },
        yaxis: {
            title: "Longitudinal Gx (mG)",
            range: [-1000, 1000],
        },
        height: 500,
        width: 500,
        margin: { l: 60, r: 20, t: 50, b: 50 },
    }, { responsive: true });
}

/** Affiche les metriques QA */
export function renderQA(containerId, qaData, metaData) {
    const el = document.getElementById(containerId);
    if (!el) return;

    const pretty = qaData.qa_pretty || {};
    const realism = qaData.qa_realism || {};
    const checklist = qaData.qa_checklist || {};
    const imu = qaData.imu_coherence || {};
    const legs = qaData.legs_summary || [];

    // Detecter si une rotation est appliquee
    const rot = (metaData || {}).device_rotation_deg || {};
    const hasRotation = (rot.roll || 0) !== 0 || (rot.pitch || 0) !== 0 || (rot.yaw || 0) !== 0;

    const isOk = realism.ok !== false;
    // Si KO mais rotation appliquee, c'est attendu
    const isExpectedKO = !isOk && hasRotation;
    const statusClass = isOk ? "success" : (isExpectedKO ? "loading" : "error");
    const statusIcon = isOk ? "\u2705" : (isExpectedKO ? "\u2139\uFE0F" : "\u26A0\uFE0F");

    // Metriques cles (filtrer les plus pertinentes)
    const metrics = checklist.metrics || realism.metrics || {};
    const keyMetrics = {
        "hz_obs": { label: "Frequence", unit: "Hz", precision: 0 },
        "v_median_mps": { label: "V mediane", unit: "km/h", precision: 1, factor: 3.6 },
        "std_ax": { label: "\u03C3 acc_x", unit: "m/s\u00B2", precision: 3 },
        "std_gz": { label: "\u03C3 gyro_z", unit: "rad/s", precision: 4 },
        "dt_median_s": { label: "\u0394t median", unit: "s", precision: 3 },
        "lat_err_mps2_med": { label: "Err laterale", unit: "m/s\u00B2", precision: 3 },
    };

    const metricsHtml = Object.entries(keyMetrics)
        .filter(([k]) => metrics[k] !== undefined)
        .map(([k, cfg]) => {
            let val = metrics[k];
            if (cfg.factor) val *= cfg.factor;
            return `<div class="metric-card">
                <div class="metric-value">${val.toFixed(cfg.precision)} ${cfg.unit}</div>
                <div class="metric-label">${cfg.label}</div>
            </div>`;
        }).join("");

    // Checks individuels (realism.checks)
    const checks = realism.checks || checklist.checks || [];
    // Noms des checks lies a la rotation (echouent normalement quand rotation appliquee)
    const rotationSensitiveChecks = ["lateral_consistency", "lat_err", "coherence"];

    const checksHtml = checks.length > 0 ? `
        <h4>Controles</h4>
        ${hasRotation ? '<p style="color:#856404;font-size:0.85em;margin-bottom:8px">\u2139\uFE0F Rotation appliqu\u00E9e (' + rot.roll + '\u00B0/' + rot.pitch + '\u00B0/' + rot.yaw + '\u00B0) \u2014 certains \u00E9checs sont attendus (le signal est volontairement d\u00E9form\u00E9 pour tester la reconstruction).</p>' : ''}
        <div class="checks-list">
            ${(Array.isArray(checks) ? checks : Object.entries(checks).map(([k, v]) => ({name: k, ...v}))).map((c) => {
                const ok = c.ok !== false && c.passed !== false;
                const name = c.label || c.name || c.check || "";
                const isRotSensitive = !ok && hasRotation && rotationSensitiveChecks.some((r) => name.toLowerCase().includes(r));
                const icon = ok ? "\u2705" : (isRotSensitive ? "\u26A0\uFE0F" : "\u274C");
                const suffix = isRotSensitive ? ' <span style="color:#856404;font-size:0.8em">(attendu avec rotation)</span>' : "";
                return `<div class="check-item ${ok ? '' : (isRotSensitive ? 'check-expected' : 'check-fail')}">${icon} ${name}${suffix}</div>`;
            }).join("")}
        </div>
    ` : "";

    // IMU coherence (seulement les metriques comprehensibles)
    const imuDisplay = {
        "rmse_ay_vs_vpsi": { label: "RMSE ay vs v\u00B7\u03C8'", precision: 4 },
        "p95_abs_resid": { label: "P95 residu", precision: 4 },
        "bad_ratio_gt_0p2": { label: "% mauvais (>0.2)", precision: 1, factor: 100, unit: "%" },
    };
    const imuHtml = Object.keys(imu).length > 0 ? `
        <h4>Coherence IMU</h4>
        <div class="metrics-grid">
            ${Object.entries(imuDisplay)
                .filter(([k]) => imu[k] !== undefined)
                .map(([k, cfg]) => {
                    let val = imu[k];
                    if (cfg.factor) val *= cfg.factor;
                    return `<div class="metric-card">
                        <div class="metric-value">${val.toFixed(cfg.precision)}${cfg.unit || ''}</div>
                        <div class="metric-label">${cfg.label}</div>
                    </div>`;
                }).join("")}
        </div>
    ` : "";

    // Legs summary
    const totalDist = legs.reduce((s, l) => s + (l.distance_m || 0), 0);
    const totalDur = legs.reduce((s, l) => s + (l.duration_s || 0), 0);
    const legsHtml = legs.length > 0 ? `
        <h4>Parcours (${legs.length} segments, ${(totalDist/1000).toFixed(1)} km, ${(totalDur/60).toFixed(0)} min)</h4>
        <table class="qa-table">
            <tr><th>Leg</th><th>Distance</th><th>Duree</th><th>V moy</th></tr>
            ${legs.map((leg, i) => `
                <tr>
                    <td>${i + 1}</td>
                    <td>${((leg.distance_m || 0) / 1000).toFixed(1)} km</td>
                    <td>${((leg.duration_s || 0) / 60).toFixed(0)} min</td>
                    <td>${((leg.mean_speed_mps || 0) * 3.6).toFixed(0)} km/h</td>
                </tr>
            `).join("")}
        </table>
    ` : "";

    let statusText = pretty.status || (isOk ? "Simulation coherente" : "Anomalies detectees");
    if (isExpectedKO) {
        statusText = `Ecarts detect\u00E9s (attendus \u2014 rotation ${rot.roll}\u00B0/${rot.pitch}\u00B0/${rot.yaw}\u00B0 appliqu\u00E9e). Le signal simule un bo\u00EEtier mal orient\u00E9, le pipeline de reconstruction doit le corriger.`;
    }

    el.innerHTML = `
        <div class="status ${statusClass}" style="margin-bottom:16px">
            ${statusIcon} ${statusText}
        </div>
        ${metricsHtml ? `<h4>Metriques cles</h4><div class="metrics-grid">${metricsHtml}</div>` : ""}
        ${checksHtml}
        ${imuHtml}
        ${legsHtml}
    `;
}

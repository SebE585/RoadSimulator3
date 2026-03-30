/**
 * RS3 Simulator — API client
 * Communique avec le backend FastAPI (api.py)
 */

const API_BASE = window.location.origin;

export async function simulate(config) {
    const resp = await fetch(`${API_BASE}/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
    });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || "Simulation failed");
    }
    return resp.json();
}

export async function getTrace(jobId, hz = 1) {
    const resp = await fetch(`${API_BASE}/results/${jobId}/trace?hz=${hz}`);
    if (!resp.ok) throw new Error("Failed to load trace");
    return resp.json();
}

export async function getSensors(jobId, maxPoints = 3000) {
    const resp = await fetch(`${API_BASE}/results/${jobId}/sensors?max_points=${maxPoints}`);
    if (!resp.ok) throw new Error("Failed to load sensors");
    return resp.json();
}

export async function getMeta(jobId) {
    const resp = await fetch(`${API_BASE}/results/${jobId}/meta`);
    if (!resp.ok) throw new Error("Failed to load meta");
    return resp.json();
}

export async function getQA(jobId) {
    const resp = await fetch(`${API_BASE}/results/${jobId}/qa`);
    if (!resp.ok) throw new Error("Failed to load QA");
    return resp.json();
}

export async function getSeverity(jobId) {
    const resp = await fetch(`${API_BASE}/results/${jobId}/severity`);
    if (!resp.ok) throw new Error("Failed to load severity");
    return resp.json();
}

export function downloadUrl(jobId, format) {
    return `${API_BASE}/download/${jobId}/${format}`;
}

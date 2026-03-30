/**
 * RS3 Simulator — Map module
 * Leaflet natif, click handlers fluides, zero iframe
 */

const DEFAULT_CENTER = [49.38, 1.25];
const DEFAULT_ZOOM = 11;

let inputMap = null;
let resultMap = null;
let stopMarkers = [];
let routeLine = null;
let resultLine = null;

/** Initialise la carte de saisie des arrets */
export function initInputMap(containerId, onClickCallback) {
    inputMap = L.map(containerId, {
        center: DEFAULT_CENTER,
        zoom: DEFAULT_ZOOM,
    });

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "&copy; OpenStreetMap contributors",
        maxZoom: 19,
    }).addTo(inputMap);

    // Click natif — fluide, pas de rerun, pas d'iframe
    inputMap.on("click", (e) => {
        const lat = Math.round(e.latlng.lat * 1e6) / 1e6;
        const lng = Math.round(e.latlng.lng * 1e6) / 1e6;
        onClickCallback(lat, lng);
    });

    return inputMap;
}

/** Ajoute un marqueur d'arret sur la carte */
export function addStopMarker(stop, index) {
    if (!inputMap) return;

    const isDepot = stop.id === "DEPOT";
    const marker = L.marker([stop.lat, stop.lon], {
        icon: L.divIcon({
            className: "stop-marker",
            html: `<div style="
                background: ${isDepot ? "#e74c3c" : "#0066CC"};
                color: white;
                width: 28px; height: 28px;
                border-radius: 50%;
                display: flex; align-items: center; justify-content: center;
                font-weight: 700; font-size: 12px;
                border: 2px solid white;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            ">${isDepot ? "D" : index}</div>`,
            iconSize: [28, 28],
            iconAnchor: [14, 14],
        }),
    }).addTo(inputMap);

    marker.bindPopup(`<b>${stop.id}</b><br>${stop.lat.toFixed(4)}, ${stop.lon.toFixed(4)}<br>Service: ${stop.service_s}s`);
    stopMarkers.push(marker);
}

/** Redessine la ligne entre les arrets */
export function updateRoute(stops) {
    if (routeLine) {
        inputMap.removeLayer(routeLine);
        routeLine = null;
    }
    if (stops.length >= 2) {
        const coords = stops.map((s) => [s.lat, s.lon]);
        routeLine = L.polyline(coords, {
            color: "#27ae60",
            weight: 3,
            opacity: 0.7,
            dashArray: "8, 8",
        }).addTo(inputMap);
    }
}

/** Efface tous les marqueurs et la route */
export function clearInputMap() {
    stopMarkers.forEach((m) => inputMap.removeLayer(m));
    stopMarkers = [];
    if (routeLine) {
        inputMap.removeLayer(routeLine);
        routeLine = null;
    }
}

/** Recree tous les marqueurs depuis la liste des stops */
export function refreshMarkers(stops) {
    clearInputMap();
    stops.forEach((s, i) => addStopMarker(s, i));
    updateRoute(stops);

    if (stops.length > 0) {
        const bounds = L.latLngBounds(stops.map((s) => [s.lat, s.lon]));
        inputMap.fitBounds(bounds, { padding: [40, 40], maxZoom: 14 });
    }
}

/** Initialise la carte des resultats */
export function initResultMap(containerId) {
    resultMap = L.map(containerId, {
        center: DEFAULT_CENTER,
        zoom: DEFAULT_ZOOM,
    });

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "&copy; OpenStreetMap contributors",
        maxZoom: 19,
    }).addTo(resultMap);

    return resultMap;
}

/** Affiche la trace simulee sur la carte resultat */
export function showTrace(traceData) {
    if (!resultMap) return;

    // Nettoyer
    if (resultLine) resultMap.removeLayer(resultLine);
    resultMap.eachLayer((layer) => {
        if (layer instanceof L.Marker) resultMap.removeLayer(layer);
    });

    // Re-ajouter le tile layer s'il a ete supprime
    let hasTiles = false;
    resultMap.eachLayer((layer) => {
        if (layer instanceof L.TileLayer) hasTiles = true;
    });
    if (!hasTiles) {
        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution: "&copy; OpenStreetMap contributors",
        }).addTo(resultMap);
    }

    const coords = traceData.coords;
    if (!coords || coords.length === 0) return;

    // Trace animee
    resultLine = L.polyline(coords, {
        color: "#0066CC",
        weight: 3,
        opacity: 0.8,
    }).addTo(resultMap);

    // Marqueurs depart/arrivee
    L.marker(coords[0], {
        icon: L.divIcon({
            className: "",
            html: '<div style="background:#27ae60;color:white;width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:14px;border:2px solid white;box-shadow:0 2px 4px rgba(0,0,0,0.3)">&#9654;</div>',
            iconSize: [24, 24],
            iconAnchor: [12, 12],
        }),
    }).addTo(resultMap);

    L.marker(coords[coords.length - 1], {
        icon: L.divIcon({
            className: "",
            html: '<div style="background:#e74c3c;color:white;width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:14px;border:2px solid white;box-shadow:0 2px 4px rgba(0,0,0,0.3)">&#9632;</div>',
            iconSize: [24, 24],
            iconAnchor: [12, 12],
        }),
    }).addTo(resultMap);

    // Zoom sur la trace
    resultMap.fitBounds(resultLine.getBounds(), { padding: [30, 30] });
}

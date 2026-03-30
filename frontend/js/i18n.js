/**
 * RS3 Simulator — Internationalisation (FR/EN)
 * Détection automatique de la locale du navigateur.
 */

const TRANSLATIONS = {
    fr: {
        // Header
        "app.title": "RoadSimulator3",
        "app.subtitle": "Simulateur télématique — Placez des arrêts, configurez, lancez",

        // Map
        "map.title": "Carte",
        "map.hint": "Cliquez sur la carte pour placer le DÉPÔT puis les clients",
        "map.btn.return": "↩ Retour DÉPÔT",
        "map.btn.remove": "✖ Supprimer dernier",
        "map.btn.clear": "Effacer tout",

        // Config
        "config.title": "Configuration",
        "config.stops": "Arrêts",
        "config.stops.empty": "Cliquez sur la carte pour ajouter le DÉPÔT puis les clients.",
        "config.gps": "📡 GPS",
        "config.gps.freq": "Fréquence (Hz)",
        "config.gps.noise": "Bruit GPS réaliste",
        "config.gps.jitter": "Jitter (m)",
        "config.gps.blackouts": "Blackouts tunnel",
        "config.gps.drift": "Drift démarrage (m)",
        "config.imu": "🔧 IMU",
        "config.imu.freq": "Fréquence (Hz)",
        "config.imu.noise": "Bruit acc (m/s²)",
        "config.imu.gyro": "Gyroscope",
        "config.rotation": "🔄 Orientation du boîtier",
        "config.profile": "🏎 Profil de conduite",
        "config.profile.custom": "Personnalisé",
        "config.profile.smooth": "Doux (éco-conduite)",
        "config.profile.normal": "Normal",
        "config.profile.aggressive": "Agressif (sportif)",
        "config.events": "⚡ Événements à injecter",
        "config.events.brake": "Freinages",
        "config.events.accel": "Accélérations",
        "config.events.bump": "Dos d'âne",
        "config.events.pothole": "Nids de poule",
        "config.events.turn": "Virages",
        "config.events.door": "Ouv. porte",
        "config.carto": "🗺 Services carto",

        // Simulation
        "sim.launch": "🚀 Lancer la simulation",
        "sim.loading": "Simulation RS3 en cours...",
        "sim.done": "Simulation terminée",
        "sim.error": "Erreur",

        // Results
        "results.title": "Résultats",
        "results.download": "📥 D0 Parquet",
        "results.tab.map": "Carte",
        "results.tab.sensors": "Capteurs",
        "results.tab.severity": "Sévérité",
        "results.tab.qa": "Qualité",

        // QA
        "qa.ok": "Simulation cohérente",
        "qa.ko": "Anomalies détectées",
        "qa.rotation.expected": "Écarts détectés (attendus — rotation {rot} appliquée). Le signal simule un boîtier mal orienté, Nostos doit le corriger.",
        "qa.rotation.note": "ℹ️ Rotation appliquée ({rot}) — certains échecs sont attendus (le signal est volontairement déformé pour tester la reconstruction).",
        "qa.rotation.suffix": "(attendu avec rotation)",
        "qa.metrics": "Métriques clés",
        "qa.checks": "Contrôles",
        "qa.imu": "Cohérence IMU",
        "qa.legs": "Parcours",

        // Charts
        "chart.acc": "Accéléromètre",
        "chart.gyro": "Gyroscope",
        "chart.gyro.off": "Gyroscope désactivé",
        "chart.speed": "Vitesse",
        "chart.severity": "Bi-histogramme",
        "chart.no_data": "Pas de données",

        // Summary
        "summary.gps_points": "Points GPS",
        "summary.duration": "Durée",
        "summary.freq": "GPS / IMU",
        "summary.gyro": "Gyroscope",
        "summary.rotation": "Rotation R/P/Y",
    },

    en: {
        "app.title": "RoadSimulator3",
        "app.subtitle": "Telematic simulator — Place stops, configure, launch",

        "map.title": "Map",
        "map.hint": "Click on the map to place the DEPOT then customers",
        "map.btn.return": "↩ Return to DEPOT",
        "map.btn.remove": "✖ Remove last",
        "map.btn.clear": "Clear all",

        "config.title": "Configuration",
        "config.stops": "Stops",
        "config.stops.empty": "Click on the map to add the DEPOT then customers.",
        "config.gps": "📡 GPS",
        "config.gps.freq": "Frequency (Hz)",
        "config.gps.noise": "Realistic GPS noise",
        "config.gps.jitter": "Jitter (m)",
        "config.gps.blackouts": "Tunnel blackouts",
        "config.gps.drift": "Start drift (m)",
        "config.imu": "🔧 IMU",
        "config.imu.freq": "Frequency (Hz)",
        "config.imu.noise": "Acc noise (m/s²)",
        "config.imu.gyro": "Gyroscope",
        "config.rotation": "🔄 Device orientation",
        "config.profile": "🏎 Driving profile",
        "config.profile.custom": "Custom",
        "config.profile.smooth": "Smooth (eco-driving)",
        "config.profile.normal": "Normal",
        "config.profile.aggressive": "Aggressive (sporty)",
        "config.events": "⚡ Events to inject",
        "config.events.brake": "Hard brakes",
        "config.events.accel": "Hard accelerations",
        "config.events.bump": "Speed bumps",
        "config.events.pothole": "Potholes",
        "config.events.turn": "Sharp turns",
        "config.events.door": "Door openings",
        "config.carto": "🗺 Map services",

        "sim.launch": "🚀 Launch simulation",
        "sim.loading": "RS3 simulation in progress...",
        "sim.done": "Simulation completed",
        "sim.error": "Error",

        "results.title": "Results",
        "results.download": "📥 D0 Parquet",
        "results.tab.map": "Map",
        "results.tab.sensors": "Sensors",
        "results.tab.severity": "Severity",
        "results.tab.qa": "Quality",

        "qa.ok": "Simulation coherent",
        "qa.ko": "Anomalies detected",
        "qa.rotation.expected": "Deviations detected (expected — rotation {rot} applied). Signal simulates a misoriented device for reconstruction testing.",
        "qa.rotation.note": "ℹ️ Rotation applied ({rot}) — some failures are expected (signal is intentionally distorted for reconstruction testing).",
        "qa.rotation.suffix": "(expected with rotation)",
        "qa.metrics": "Key metrics",
        "qa.checks": "Checks",
        "qa.imu": "IMU coherence",
        "qa.legs": "Route",

        "chart.acc": "Accelerometer",
        "chart.gyro": "Gyroscope",
        "chart.gyro.off": "Gyroscope disabled",
        "chart.speed": "Speed",
        "chart.severity": "Bi-histogram",
        "chart.no_data": "No data",

        "summary.gps_points": "GPS points",
        "summary.duration": "Duration",
        "summary.freq": "GPS / IMU",
        "summary.gyro": "Gyroscope",
        "summary.rotation": "Rotation R/P/Y",
    },
};

let _locale = "fr";

/** Détecte la locale du navigateur */
export function detectLocale() {
    const nav = navigator.language || navigator.userLanguage || "fr";
    _locale = nav.startsWith("fr") ? "fr" : "en";
    return _locale;
}

/** Retourne la locale courante */
export function getLocale() {
    return _locale;
}

/** Change la locale */
export function setLocale(locale) {
    _locale = locale === "fr" ? "fr" : "en";
}

/** Traduit une clé, avec remplacement de variables {var} */
export function t(key, vars) {
    const dict = TRANSLATIONS[_locale] || TRANSLATIONS.fr;
    let text = dict[key] || TRANSLATIONS.fr[key] || key;
    if (vars) {
        for (const [k, v] of Object.entries(vars)) {
            text = text.replace(`{${k}}`, v);
        }
    }
    return text;
}

/** Applique les traductions à tous les éléments avec data-i18n */
export function applyTranslations() {
    document.querySelectorAll("[data-i18n]").forEach((el) => {
        const key = el.getAttribute("data-i18n");
        const translated = t(key);
        // Pour les boutons avec icônes HTML, préserver l'icône
        if (el.tagName === "BUTTON" || el.tagName === "A") {
            // Garder le premier caractère emoji s'il existe
            const current = el.textContent.trim();
            const emoji = current.match(/^[\u{1F000}-\u{1FFFF}\u{2600}-\u{27BF}\u{FE00}-\u{FEFF}↩✖]/u);
            el.textContent = translated;
        } else {
            el.textContent = translated;
        }
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
        el.placeholder = t(el.getAttribute("data-i18n-placeholder"));
    });
    // Mettre à jour le titre de la page
    document.title = `RS3 — ${t("app.title")}`;
}

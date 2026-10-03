// GeoMiner ML Frontend Logic

// Global State
let map = null;
let currentBasinData = null;
let currentPredictions = null;
let activeOverlayLayer = 'geology'; // 'geology' or 'prospectivity'
let gridLayers = [];
let faultLayers = [];
let depositLayers = [];
let chartImportance = null;
let chartSpectral = null;

const API_BASE = window.location.origin;

// Helper: Convert spatial meter offset to Lat/Lng (origin: Latitude 45.0, Longitude 10.0)
function meterToLatLng(x, y) {
    const latOrigin = 45.0;
    const lngOrigin = 10.0;
    const lat = latOrigin + (y / 111300.0);
    const lng = lngOrigin + (x / (111300.0 * Math.cos(latOrigin * Math.PI / 180.0)));
    return [lat, lng];
}

// Get Lithology Color
function getLithologyColor(lithology) {
    switch (lithology) {
        case 0: return '#374151'; // Shale (Gray)
        case 1: return '#0d9488'; // Limestone (Teal)
        case 2: return '#991b1b'; // Granite (Dark Red)
        case 3: return '#1e3a8a'; // Basalt (Dark Blue)
        default: return '#1f2937';
    }
}

// Get Lithology Name
function getLithologyName(lithology) {
    switch (lithology) {
        case 0: return 'Shale / Sandstone';
        case 1: return 'Limestone';
        case 2: return 'Granitic Batholith';
        case 3: return 'Mafic Volcanic (Basalt)';
        default: return 'Unknown';
    }
}

// Get Prospectivity Color Scale
function getProspectivityColor(p) {
    if (p < 0.2) return 'rgba(88, 28, 135, 0.4)';  // Deep Purple
    if (p < 0.4) return 'rgba(99, 102, 241, 0.5)';  // Indigo
    if (p < 0.6) return 'rgba(16, 185, 129, 0.6)';  // Emerald Green
    if (p < 0.8) return 'rgba(249, 115, 22, 0.75)'; // Orange
    return 'rgba(234, 179, 8, 0.9)';                // Bright Yellow (High Prospectivity)
}

// Initialize Leaflet Map
function initMap() {
    if (map) return;
    
    // Map centered at the simulated geological coordinates
    const mapCenter = meterToLatLng(25000, 25000);
    map = L.map('map', {
        zoomControl: true,
        attributionControl: false
    }).setView(mapCenter, 11);
    
    // Add CartoDB Dark Matter tiles
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        maxZoom: 19
    }).addTo(map);
}

// Render Basin Data on Map
function renderBasin(basinData) {
    // Clear previous layers
    gridLayers.forEach(l => map.removeLayer(l));
    faultLayers.forEach(l => map.removeLayer(l));
    depositLayers.forEach(l => map.removeLayer(l));
    gridLayers = [];
    faultLayers = [];
    depositLayers = [];
    
    const cellSize = basinData.cell_size_m;
    const halfCell = cellSize / 2.0;
    
    // Use canvas renderer for high performance with 2500 grid cells
    const canvasRenderer = L.canvas();
    
    // 1. Draw Grid Cells (Geology / Prospectivity)
    basinData.grid.forEach((cell, idx) => {
        const pCenter = meterToLatLng(cell.x, cell.y);
        
        // Calculate cell bounding box corners
        const pMin = meterToLatLng(cell.x - halfCell, cell.y - halfCell);
        const pMax = meterToLatLng(cell.x + halfCell, cell.y + halfCell);
        const bounds = [pMin, pMax];
        
        // Base geological color
        const baseColor = getLithologyColor(cell.lithology);
        
        const rect = L.rectangle(bounds, {
            renderer: canvasRenderer,
            fillColor: baseColor,
            fillOpacity: 0.6,
            color: 'rgba(255, 255, 255, 0.05)',
            weight: 0.5,
            className: 'leaflet-interactive-cell'
        });
        
        // Keep cross-references to update color maps easily
        cell.mapRect = rect;
        cell.index = idx;
        
        // Hover readouts on HUD
        rect.on('mouseover', () => {
            showHUD(cell);
        });
        
        rect.on('mouseout', () => {
            hideHUD();
        });
        
        rect.addTo(map);
        gridLayers.push(rect);
    });
    
    // 2. Draw Fault Lines
    basinData.faults.forEach(fault => {
        const latlngs = fault.map(pt => meterToLatLng(pt.x, pt.y));
        const polyline = L.polyline(latlngs, {
            color: '#ef4444',
            weight: 3,
            dashArray: '6, 8',
            opacity: 0.85
        }).addTo(map);
        faultLayers.push(polyline);
    });
    
    // 3. Draw Known Deposits
    basinData.deposits.forEach(dep => {
        const center = meterToLatLng(dep.x, dep.y);
        const marker = L.circleMarker(center, {
            radius: 6.5,
            fillColor: '#fbbf24',
            color: '#ffffff',
            weight: 1.5,
            fillOpacity: 0.95,
            className: 'animate-pulse'
        });
        
        marker.bindPopup(`
            <div class="text-xs font-semibold p-1">
                <span class="text-amber-500 font-bold uppercase block mb-1">Occurence Verified</span>
                Copper: ${dep.cu_ppm.toFixed(1)} ppm<br/>
                Gold: ${dep.au_ppb.toFixed(2)} ppb
            </div>
        `, { closeButton: false });
        
        marker.addTo(map);
        depositLayers.push(marker);
    });
}

// Display HUD with hover details
const hud = document.getElementById('hud');
function showHUD(cell) {
    hud.style.display = 'block';
    
    document.getElementById('hud-east').innerText = cell.easting.toLocaleString(undefined, {maximumFractionDigits: 0});
    document.getElementById('hud-north').innerText = cell.northing.toLocaleString(undefined, {maximumFractionDigits: 0});
    document.getElementById('hud-fault').innerText = `${(cell.fault_dist / 1000.0).toFixed(2)} km`;
    document.getElementById('hud-mag').innerText = `${cell.mag_anomaly.toFixed(1)} nT`;
    document.getElementById('hud-grav').innerText = `${cell.grav_anomaly.toFixed(2)} mGal`;
    document.getElementById('hud-litho').innerText = getLithologyName(cell.lithology);
    document.getElementById('hud-cu').innerText = `${cell.cu_ppm.toFixed(1)} ppm`;
    document.getElementById('hud-au').innerText = `${cell.au_ppb.toFixed(2)} ppb`;
    
    const probSpan = document.getElementById('hud-prob');
    if (currentPredictions && currentPredictions[cell.index] !== undefined) {
        const probVal = currentPredictions[cell.index];
        probSpan.innerText = `${(probVal * 100.0).toFixed(1)}%`;
        if (probVal > 0.7) {
            probSpan.className = "font-extrabold text-md text-amber-400 glow-text-accent";
        } else if (probVal > 0.4) {
            probSpan.className = "font-extrabold text-md text-emerald-400 glow-text-secondary";
        } else {
            probSpan.className = "font-extrabold text-md text-gray-500";
        }
    } else {
        probSpan.innerText = 'Untrained';
        probSpan.className = "font-semibold text-gray-500 text-sm italic";
    }
}

function hideHUD() {
    // Keep HUD open but don't hide it instantly so it is easier to read, or hide it
    // Let's keep it visible so it doesn't flicker, just update content when hovering.
}

// Fetch current simulated basin data from FastAPI
async function fetchBasinData() {
    try {
        const res = await fetch(`${API_BASE}/api/simulator/data`);
        if (!res.ok) throw new Error("Failed to fetch basin data");
        currentBasinData = await res.json();
        
        document.getElementById('stat-deposits').innerText = currentBasinData.deposits.length;
        renderBasin(currentBasinData);
        
        // Reset predictions
        currentPredictions = null;
        document.getElementById('layer-heat').disabled = true;
        document.getElementById('layer-heat').className = "px-2.5 py-1 bg-white/5 text-gray-600 font-medium cursor-not-allowed";
        toggleLayer('geology');
    } catch (e) {
        console.error(e);
        alert("Failed to connect to the backend server. Make sure main.py is running!");
    }
}

// Toggle Display Layer between Geology and ML Heatmap
function toggleLayer(layer) {
    activeOverlayLayer = layer;
    const btnGeology = document.getElementById('layer-none');
    const btnHeat = document.getElementById('layer-heat');
    
    if (layer === 'geology') {
        btnGeology.className = "px-2.5 py-1 bg-indigo-500/20 text-white font-medium border-r border-white/10";
        btnHeat.className = currentPredictions ? "px-2.5 py-1 bg-white/5 text-gray-400 hover:text-white font-medium" : "px-2.5 py-1 bg-white/5 text-gray-600 font-medium cursor-not-allowed";
        
        if (currentBasinData) {
            currentBasinData.grid.forEach(cell => {
                if (cell.mapRect) {
                    cell.mapRect.setStyle({
                        fillColor: getLithologyColor(cell.lithology),
                        fillOpacity: 0.6
                    });
                }
            });
        }
    } else if (layer === 'prospectivity' && currentPredictions) {
        btnGeology.className = "px-2.5 py-1 bg-white/5 text-gray-400 hover:text-white font-medium border-r border-white/10";
        btnHeat.className = "px-2.5 py-1 bg-indigo-500/20 text-white font-medium";
        
        currentBasinData.grid.forEach((cell, idx) => {
            if (cell.mapRect && currentPredictions[idx] !== undefined) {
                const prob = currentPredictions[idx];
                cell.mapRect.setStyle({
                    fillColor: getProspectivityColor(prob),
                    fillOpacity: 0.8
                });
            }
        });
    }
}

// Train ML Model and Map
async function trainModel() {
    const btnTrain = document.getElementById('btn-train');
    const modelType = document.getElementById('select-model').value;
    
    // Get Selected Features
    const features = [];
    document.querySelectorAll('input[name="feature"]:checked').forEach(cb => {
        features.push(cb.value);
    });
    
    if (features.length === 0) {
        alert("Please select at least one GIS predictor layer!");
        return;
    }
    
    // Get Hyperparams
    const hyperparams = {};
    if (modelType === 'random_forest') {
        hyperparams.n_estimators = document.getElementById('input-rf-trees').value;
        hyperparams.max_depth = document.getElementById('input-rf-depth').value;
    } else if (modelType === 'gradient_boosting') {
        hyperparams.n_estimators = document.getElementById('input-gb-trees').value;
        hyperparams.learning_rate = document.getElementById('input-gb-lr').value;
    } else if (modelType === 'svm') {
        hyperparams.C = document.getElementById('input-svm-c').value;
    }
    
    btnTrain.disabled = true;
    btnTrain.innerHTML = `<span class="spinner"></span> Mapping Mineral Potential...`;
    
    try {
        // 1. Train Model
        const trainRes = await fetch(`${API_BASE}/api/model/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_type: modelType,
                features: features,
                hyperparams: hyperparams
            })
        });
        
        if (!trainRes.ok) throw new Error("Model training failed");
        const results = await trainRes.json();
        
        // 2. Predict prospectivity grid
        const predRes = await fetch(`${API_BASE}/api/model/predict`, { method: 'POST' });
        if (!predRes.ok) throw new Error("Prospectivity mapping failed");
        const predictions = await predRes.json();
        currentPredictions = predictions.predictions;
        
        // 3. Update Metrics UI
        document.getElementById('metrics-placeholder').classList.add('hidden');
        document.getElementById('metrics-container').classList.remove('hidden');
        
        document.getElementById('metric-acc').innerText = `${(results.metrics.accuracy * 100.0).toFixed(1)}%`;
        document.getElementById('metric-f1').innerText = results.metrics.f1_score.toFixed(3);
        document.getElementById('metric-auc').innerText = results.metrics.auc.toFixed(3);
        
        // Enable Prospectivity overlay layer
        const btnHeat = document.getElementById('layer-heat');
        btnHeat.disabled = false;
        btnHeat.className = "px-2.5 py-1 bg-white/5 text-gray-400 hover:text-white font-medium";
        
        // Render Feature Importance Chart
        renderFeatureImportance(results.feature_importances);
        
        // Switch to Prospectivity Overlay
        toggleLayer('prospectivity');
        
    } catch (e) {
        console.error(e);
        alert("ML execution failed: " + e.message);
    } finally {
        btnTrain.disabled = false;
        btnTrain.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" class="h-4.5 w-4.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                <path stroke-linecap="round" stroke-linejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Train & Map Prospectivity
        `;
    }
}

// Render Feature Importance Chart
function renderFeatureImportance(importances) {
    const ctx = document.getElementById('chart-importance').getContext('2d');
    
    if (chartImportance) {
        chartImportance.destroy();
    }
    
    // Feature clean names mapping
    const cleanNames = {
        'fault_dist': 'Fault Distance',
        'mag_anomaly': 'Magnetics',
        'grav_anomaly': 'Gravity',
        'cu_ppm': 'Copper Assay',
        'au_ppb': 'Gold Assay',
        'hydroxyl_idx': 'Hydroxyl Clay',
        'fe_oxide_idx': 'Iron Oxide'
    };
    
    const labels = importances.map(item => cleanNames[item.feature] || item.feature);
    const data = importances.map(item => item.importance * 100.0);
    
    chartImportance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Importance weight (%)',
                data: data,
                backgroundColor: 'rgba(99, 102, 241, 0.75)',
                borderColor: 'rgb(99, 102, 241)',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af', font: { size: 10 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#f3f4f6', font: { size: 10, weight: 'bold' } }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// Interactive Spectroscopy Lab
const sliders = {
    alunite: document.getElementById('slide-alunite'),
    kaolinite: document.getElementById('slide-kaolinite'),
    hematite: document.getElementById('slide-hematite'),
    chalco: document.getElementById('slide-chalco')
};

// Bind sliders values update
Object.keys(sliders).forEach(key => {
    sliders[key].addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        document.getElementById(`pct-${key}`).innerText = `${val}%`;
        balanceSliders(key, val);
    });
});

// Balance sliders so they sum up to <= 100%
function balanceSliders(activeKey, activeVal) {
    const keys = Object.keys(sliders);
    let totalOthers = 0;
    keys.forEach(k => {
        if (k !== activeKey) totalOthers += parseInt(sliders[k].value);
    });
    
    if (activeVal + totalOthers > 100) {
        const scale = (100 - activeVal) / (totalOthers || 1);
        keys.forEach(k => {
            if (k !== activeKey) {
                const scaledVal = Math.round(parseInt(sliders[k].value) * scale);
                sliders[k].value = scaledVal;
                document.getElementById(`pct-${k}`).innerText = `${scaledVal}%`;
            }
        });
    }
}

// Acquire spectrum and classify
async function acquireSpectrum() {
    const composition = {
        'Alunite': parseInt(sliders.alunite.value) / 100.0,
        'Kaolinite': parseInt(sliders.kaolinite.value) / 100.0,
        'Hematite': parseInt(sliders.hematite.value) / 100.0,
        'Chalcopyrite': parseInt(sliders.chalco.value) / 100.0
    };
    
    // Add quartz as background to fill remaining
    const totalMix = Object.values(composition).reduce((a, b) => a + b, 0);
    if (totalMix < 1.0) {
        composition['Quartz'] = 1.0 - totalMix;
    }
    
    const btnAcquire = document.getElementById('btn-acquire');
    btnAcquire.disabled = true;
    btnAcquire.innerText = 'Scanning Sample...';
    
    try {
        // 1. Generate noisy mixed spectrum from API
        const genRes = await fetch(`${API_BASE}/api/spectroscopy/generate_sample`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ composition: composition })
        });
        
        if (!genRes.ok) throw new Error("Spectral generation failed");
        const specData = await genRes.json();
        
        // 2. Classify spectrum using SAM API
        const classRes = await fetch(`${API_BASE}/api/spectroscopy/classify`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ spectrum: specData.spectrum })
        });
        
        if (!classRes.ok) throw new Error("Spectral classification failed");
        const matchResults = await classRes.json();
        
        // Show container
        document.getElementById('spectral-results-container').classList.remove('hidden');
        
        // Render Spectral Curve Chart
        renderSpectralChart(specData.wavelengths, specData.spectrum);
        
        // Update SAM results text
        document.getElementById('sam-best-mineral').innerText = matchResults.best_match;
        document.getElementById('sam-angle').innerText = `${matchResults.spectral_angle_deg.toFixed(2)}°`;
        document.getElementById('sam-score').innerText = `${(matchResults.match_score * 100.0).toFixed(1)}%`;
        
    } catch (e) {
        console.error(e);
        alert("Spectroscopy scanning failed: " + e.message);
    } finally {
        btnAcquire.disabled = false;
        btnAcquire.innerText = 'Acquire Reflectance Curve';
    }
}

// Render Spectral Chart
function renderSpectralChart(wavelengths, spectrum) {
    const ctx = document.getElementById('chart-spectral').getContext('2d');
    
    if (chartSpectral) {
        chartSpectral.destroy();
    }
    
    chartSpectral = new Chart(ctx, {
        type: 'line',
        data: {
            labels: wavelengths,
            datasets: [{
                label: 'Sample Reflectance',
                data: spectrum,
                borderColor: 'rgb(245, 158, 11)',
                borderWidth: 2,
                pointRadius: 0,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { 
                        color: '#9ca3af',
                        callback: function(val, index) {
                            return index % 4 === 0 ? this.getLabelForValue(val) : '';
                        },
                        font: { size: 9 }
                    },
                    title: { display: true, text: 'Wavelength (nm)', color: '#9ca3af', font: { size: 9 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af', font: { size: 9 } },
                    min: 0,
                    max: 1.0,
                    title: { display: true, text: 'Reflectance', color: '#9ca3af', font: { size: 9 } }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// Bind Select Model Events to toggle Hyperparams Sliders
document.getElementById('select-model').addEventListener('change', (e) => {
    const type = e.target.value;
    document.getElementById('hyperparams-rf').classList.add('hidden');
    document.getElementById('hyperparams-gb').classList.add('hidden');
    document.getElementById('hyperparams-svm').classList.add('hidden');
    
    if (type === 'random_forest') {
        document.getElementById('hyperparams-rf').classList.remove('hidden');
    } else if (type === 'gradient_boosting') {
        document.getElementById('hyperparams-gb').classList.remove('hidden');
    } else if (type === 'svm') {
        document.getElementById('hyperparams-svm').classList.remove('hidden');
    }
});

// Bind sliders value display
document.getElementById('input-rf-trees').addEventListener('input', (e) => {
    document.getElementById('val-rf-trees').innerText = e.target.value;
});
document.getElementById('input-rf-depth').addEventListener('input', (e) => {
    document.getElementById('val-rf-depth').innerText = e.target.value;
});
document.getElementById('input-gb-trees').addEventListener('input', (e) => {
    document.getElementById('val-gb-trees').innerText = e.target.value;
});
document.getElementById('input-gb-lr').addEventListener('input', (e) => {
    document.getElementById('val-gb-lr').innerText = parseFloat(e.target.value).toFixed(2);
});
document.getElementById('input-svm-c').addEventListener('input', (e) => {
    document.getElementById('val-svm-c').innerText = parseFloat(e.target.value).toFixed(1);
});

// Bind buttons
document.getElementById('btn-train').addEventListener('click', trainModel);
document.getElementById('btn-acquire').addEventListener('click', acquireSpectrum);

// Re-generate spatial basin trigger
document.getElementById('btn-re-simulate').addEventListener('click', async () => {
    const btn = document.getElementById('btn-re-simulate');
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner" style="width: 14px; height: 14px; border-width: 2px;"></span> Simulating Basin...`;
    
    try {
        const res = await fetch(`${API_BASE}/api/simulator/generate`, { method: 'POST' });
        if (!res.ok) throw new Error("Re-simulation failed");
        
        // Randomize the model name in header for premium feel
        const models = ['Porphyry Cu-Au', 'Orogenic Gold', 'Sediment-Hosted Cu', 'VMS Zinc-Lead'];
        const randomModel = models[Math.floor(Math.random() * models.length)];
        document.getElementById('txt-basin-model').innerText = randomModel;
        
        await fetchBasinData();
    } catch (e) {
        console.error(e);
        alert("Failed to simulate new basin: " + e.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                <path stroke-linecap="round" stroke-linejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 1121.21 8H18.5" />
            </svg>
            Generate New Basin
        `;
    }
});

document.getElementById('layer-none').addEventListener('click', () => toggleLayer('geology'));
document.getElementById('layer-heat').addEventListener('click', () => toggleLayer('prospectivity'));

// Startup Initialization
window.addEventListener('DOMContentLoaded', async () => {
    initMap();
    await fetchBasinData();
});

# Remote Sensing & Mineral Prospectivity Machine Learning Platform (GeoMiner ML)

GeoMiner ML is an interactive, full-stack spatial data science and machine learning dashboard designed to model geological basins, analyze geophysics/geochemistry, run predictive ML classification algorithms, and inspect mineral reflectance curves using a simulated spectroscopy laboratory.

## Platform Features

1. **GIS Spatial Map (Leaflet)**:
   - Dynamic grid rendering (50x50 km basin with 2,500 grid cells).
   - Real-time HUD overlay on hover showing coordinates (Easting/Northing), structural faults, geophysics (Magnetics/Gravity), geochemistry (assays for Cu/Au), and remote sensing alteration indices (Hydroxyl, Fe-Oxides).
   - Layer toggles between Base Geology and ML-derived Prospectivity Heatmap.

2. **Machine Learning Predictor**:
   - Supported algorithms: Random Forest Classifier, Gradient Boosting Classifier, Support Vector Machine (RBF).
   - Interactive hyperparameter sliders for tuning the estimators, depth, learning rate, and regularization.
   - GIS Predictor Layer selection checkboxes.
   - Comprehensive metrics visualization: overall Accuracy, F1-Score, and AUC.
   - Bar chart of Feature Importance mapping.

3. **Spectroscopy Laboratory**:
   - Blend mineral mixtures: Alunite (hydrothermal clay), Kaolinite (clay), Hematite (iron oxide), Chalcopyrite (sulfide copper ore), and Quartz (base background).
   - Spectral generation with Gaussian noise.
   - Spectral Angle Mapper (SAM) classification matching the sample to database reference curves, outputting the best match, spectral angle in degrees, confidence, and estimated composition.

## Technology Stack

- **Backend**: Python 3.8+, FastAPI, Uvicorn, Scikit-Learn, NumPy, Pandas, Pydantic
- **Frontend**: HTML5, Vanilla JavaScript, Tailwind CSS (CDN), Leaflet.js (GIS Map), Chart.js (Charts & Visualizations)

## Directory Structure

```
remote-sensing-ml/
├── backend/
│   ├── main.py              # FastAPI server & route handlers
│   ├── simulator.py         # Geological & GIS layer simulator
│   ├── ml_pipeline.py       # Scikit-learn training & prediction pipeline
│   ├── spectral_library.py   # Reference library & SAM classifier
│   ├── test_backend.py      # Integration test suite
│   └── static/              # Frontend files
│       ├── index.html       # Dashboard HTML structure
│       ├── styles.css       # Custom glassmorphic styling
│       └── app.js           # Frontend logic, map & charts rendering
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Setup & Running the Application

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Install Dependencies
Navigate to the project root and run:
```bash
pip install -r requirements.txt
```

### 3. Run Integration Tests (Optional)
To verify that all simulated layers, ML modeling, and spectroscopy code works properly:
```bash
python backend/test_backend.py
```

### 4. Start the Server
Run the FastAPI development server:
```bash
python backend/main.py
```
By default, the server will start at `http://127.0.0.1:8000`.

### 5. Access the Web Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:8000/
```
The static files will be served directly by FastAPI.

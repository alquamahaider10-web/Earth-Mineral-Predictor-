from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os

from simulator import GeologicalSimulator
from ml_pipeline import MineralProspectivityPipeline
from spectral_library import SpectralLibrary

app = FastAPI(title="GeoMiner ML Backend", description="GIS and Machine Learning server for mineralogy and prospectivity mapping")

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact origins (e.g., http://localhost:5173)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for current session
session_data: Dict[str, Any] = {
    "simulator": None,
    "dataset": None,
    "pipeline": None
}

# Spectral library helper
spectral_lib = SpectralLibrary()

class TrainRequest(BaseModel):
    model_type: str
    features: List[str]
    hyperparams: Optional[Dict[str, Any]] = None

class SpectrumMixRequest(BaseModel):
    composition: Dict[str, float]
    noise_level: Optional[float] = 0.015

class SpectrumClassifyRequest(BaseModel):
    spectrum: List[float]

@app.on_event("startup")
def startup_event():
    # Pre-generate dataset on startup
    session_data["simulator"] = GeologicalSimulator(grid_size=50)
    session_data["dataset"] = session_data["simulator"].simulate_basin()
    session_data["pipeline"] = MineralProspectivityPipeline()

@app.post("/api/simulator/generate")
def generate_new_basin(grid_size: int = 50):
    try:
        session_data["simulator"] = GeologicalSimulator(grid_size=grid_size)
        session_data["dataset"] = session_data["simulator"].simulate_basin()
        return {
            "status": "success",
            "message": f"Generated new basin with {grid_size}x{grid_size} grid cells",
            "faults_count": len(session_data["dataset"]["faults"]),
            "deposits_count": len(session_data["dataset"]["deposits"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/simulator/data")
def get_current_basin_data():
    if session_data["dataset"] is None:
        raise HTTPException(status_code=400, detail="No dataset has been generated yet.")
    # Exclude raw grids if they are too heavy, or return the whole payload
    # Since it's only 2500 points, returning the full dataset is very fast (~1MB)
    return session_data["dataset"]

@app.post("/api/model/train")
def train_model(payload: TrainRequest):
    if session_data["dataset"] is None:
        raise HTTPException(status_code=400, detail="No dataset available for training. Call /api/simulator/generate first.")
    
    try:
        pipeline = MineralProspectivityPipeline()
        results = pipeline.train(
            df_data=session_data["dataset"]["grid"],
            features=payload.features,
            model_type=payload.model_type,
            hyperparams=payload.hyperparams or {}
        )
        # Cache pipeline
        session_data["pipeline"] = pipeline
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model/predict")
def predict_prospectivity():
    pipeline = session_data["pipeline"]
    dataset = session_data["dataset"]
    
    if pipeline is None or pipeline.model is None:
        raise HTTPException(status_code=400, detail="Model is not trained yet. Call /api/model/train first.")
    if dataset is None:
        raise HTTPException(status_code=400, detail="Dataset not found.")
        
    try:
        predictions = pipeline.predict_grid(dataset["grid"])
        return {
            "status": "success",
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/spectroscopy/library")
def get_spectral_library():
    return spectral_lib.get_library_data()

@app.post("/api/spectroscopy/generate_sample")
def generate_sample_spectrum(payload: SpectrumMixRequest):
    try:
        # Ensure noise_level is a float (avoid Optional[float] type issues)
        noise_level = payload.noise_level if payload.noise_level is not None else 0.015
        sample = spectral_lib.generate_mixture_spectrum(payload.composition, noise_level)
        return {
            "wavelengths": spectral_lib.wavelengths.tolist(),
            "spectrum": sample
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/spectroscopy/classify")
def classify_sample_spectrum(payload: SpectrumClassifyRequest):
    try:
        classification = spectral_lib.classify_sam(payload.spectrum)
        return classification
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

script_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(script_dir, "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

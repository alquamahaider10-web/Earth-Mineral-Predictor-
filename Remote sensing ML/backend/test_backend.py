from fastapi.testclient import TestClient
import numpy as np
import sys

from main import app, session_data

client = TestClient(app)

def run_tests():
    print("=== GeoMiner ML Backend Integration Tests ===")
    
    with TestClient(app) as client:
        # 1. Test startup/initialization
        # TestClient automatically triggers startup events when used as a context manager
        print("Testing startup initialization...")
        if session_data["dataset"] is None:
            print("FAIL: Startup event failed to pre-generate dataset.")
            sys.exit(1)
        print(f"SUCCESS: Pre-generated {len(session_data['dataset']['grid'])} grid points.")
        print(f"SUCCESS: Pre-generated {len(session_data['dataset']['deposits'])} deposits.")
        
        # 2. Test Get Basin Data
        print("Testing /api/simulator/data endpoint...")
        response = client.get("/api/simulator/data")
        assert response.status_code == 200
        data = response.json()
        assert "grid" in data
        assert "faults" in data
        assert "deposits" in data
        print("SUCCESS: /api/simulator/data responded successfully.")

        # 3. Test Generate New Basin
        print("Testing /api/simulator/generate endpoint...")
        response = client.post("/api/simulator/generate?grid_size=40")
        assert response.status_code == 200
        res = response.json()
        assert res["status"] == "success"
        assert len(session_data["dataset"]["grid"]) == 1600
        print("SUCCESS: /api/simulator/generate created 40x40 grid.")

        # 4. Test Model Training
        print("Testing /api/model/train endpoint...")
        train_payload = {
            "model_type": "random_forest",
            "features": ["fault_dist", "mag_anomaly", "grav_anomaly", "cu_ppm", "au_ppb"],
            "hyperparams": {"n_estimators": 50, "max_depth": 5}
        }
        response = client.post("/api/model/train", json=train_payload)
        assert response.status_code == 200
        res = response.json()
        assert "metrics" in res
        assert "feature_importances" in res
        assert "roc_curve" in res
        print(f"SUCCESS: Model trained. Accuracy: {res['metrics']['accuracy']:.3f}, AUC: {res['metrics']['auc']:.3f}")
        
        # 5. Test Model Predictions
        print("Testing /api/model/predict endpoint...")
        response = client.post("/api/model/predict")
        assert response.status_code == 200
        res = response.json()
        assert res["status"] == "success"
        assert len(res["predictions"]) == 1600
        print(f"SUCCESS: Prospectivity predictions computed. Range: [{min(res['predictions']):.3f}, {max(res['predictions']):.3f}]")

        # 6. Test Spectral Library
        print("Testing /api/spectroscopy/library...")
        response = client.get("/api/spectroscopy/library")
        assert response.status_code == 200
        lib_data = response.json()
        assert "wavelengths" in lib_data
        assert "Alunite" in lib_data["minerals"]
        print("SUCCESS: Loaded spectral library curves.")

        # 7. Test Spectral Identification
        print("Testing /api/spectroscopy/generate_sample and classify...")
        mix_payload = {
            "composition": {"Alunite": 0.6, "Kaolinite": 0.4},
            "noise_level": 0.01
        }
        response = client.post("/api/spectroscopy/generate_sample", json=mix_payload)
        assert response.status_code == 200
        sample_data = response.json()
        
        classify_payload = {
            "spectrum": sample_data["spectrum"]
        }
        response = client.post("/api/spectroscopy/classify", json=classify_payload)
        assert response.status_code == 200
        match_data = response.json()
        assert match_data["best_match"] in ["Alunite", "Kaolinite"]
        assert "estimated_composition" in match_data
        print(f"SUCCESS: Spectral analysis matched: {match_data['best_match']} (Confidence: {match_data['match_score']*100.0:.1f}%)")

        print("\nALL TEST SUITES PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()

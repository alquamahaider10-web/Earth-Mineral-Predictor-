import numpy as np
from typing import Dict, List, Tuple

class SpectralLibrary:
    def __init__(self):
        # Wavelengths from 400nm to 2500nm at 10nm spacing (211 spectral bands)
        self.wavelengths = np.arange(400, 2510, 10)
        self.minerals = {}
        self._init_reference_library()

    def _init_reference_library(self):
        """Initializes diagnostic reflectance curves for standard minerals."""
        w = self.wavelengths
        
        # 1. Alunite (Hydrothermal alteration clay - absorption at ~1480nm and ~2170nm)
        # Reflectance climbs from UV, drops at 1480, climbs, deep drop at 2170, rises
        alunite = 0.5 + 0.3 * np.sin((w - 400) / 400)
        alunite -= 0.4 * np.exp(-((w - 1480)/60)**2)  # 1480nm absorption
        alunite -= 0.5 * np.exp(-((w - 2170)/50)**2)  # 2170nm absorption
        self.minerals['Alunite'] = np.clip(alunite, 0.05, 0.9)

        # 2. Kaolinite (Clay alteration - doublet absorption at ~2160nm and ~2200nm)
        kaolinite = 0.4 + 0.3 * (w / 2500.0)
        kaolinite -= 0.3 * np.exp(-((w - 1400)/70)**2)
        # Doublet:
        kaolinite -= 0.35 * np.exp(-((w - 2160)/30)**2)
        kaolinite -= 0.4 * np.exp(-((w - 2200)/30)**2)
        self.minerals['Kaolinite'] = np.clip(kaolinite, 0.05, 0.9)

        # 3. Illite (Sericite clay alteration - sharp single absorption at 2200nm)
        illite = 0.3 + 0.4 * np.sin((w - 300) / 600)
        illite -= 0.25 * np.exp(-((w - 1400)/50)**2)
        illite -= 0.45 * np.exp(-((w - 2200)/40)**2)
        illite -= 0.15 * np.exp(-((w - 2350)/60)**2)
        self.minerals['Illite'] = np.clip(illite, 0.05, 0.9)

        # 4. Hematite (Iron oxide - strong absorption in blue, steep slope in VNIR, absorption at ~860nm)
        hematite = 0.15 + 0.65 * (1.0 / (1.0 + np.exp(-(w - 600)/100)))  # Steep edge at 600nm
        hematite -= 0.25 * np.exp(-((w - 860)/100)**2)  # Broad Fe3+ crystal field absorption
        self.minerals['Hematite'] = np.clip(hematite, 0.02, 0.8)

        # 5. Chalcopyrite (Sulfide mineral - low reflectance, metallic flat curve, minor weathering features)
        chalcopyrite = 0.12 * np.ones_like(w) + 0.05 * np.sin(w / 1000.0)
        chalcopyrite -= 0.04 * np.exp(-((w - 1000)/200)**2)
        self.minerals['Chalcopyrite'] = np.clip(chalcopyrite, 0.01, 0.3)

        # 6. Quartz (Background - high, flat, minor features)
        quartz = 0.8 * np.ones_like(w) + np.random.normal(0, 0.01, w.shape)
        quartz -= 0.05 * np.exp(-((w - 1900)/100)**2)  # Trace water absorption
        self.minerals['Quartz'] = np.clip(quartz, 0.05, 0.95)

    def get_library_data(self) -> Dict:
        """Returns the reference library spectra for plotting."""
        return {
            'wavelengths': self.wavelengths.tolist(),
            'minerals': {name: curve.tolist() for name, curve in self.minerals.items()}
        }

    def generate_mixture_spectrum(self, composition: Dict[str, float], noise_level: float = 0.02) -> List[float]:
        """Generates a synthetic reflectance spectrum for a mixed rock sample with noise."""
        total_weight = sum(composition.values())
        if total_weight == 0:
            return list(self.minerals['Quartz'])
            
        mixed_spectrum = np.zeros_like(self.wavelengths, dtype=float)
        for name, pct in composition.items():
            if name in self.minerals:
                mixed_spectrum += self.minerals[name] * (pct / total_weight)
                
        # Add random Gaussian noise
        noise = np.random.normal(0, noise_level, self.wavelengths.shape)
        mixed_spectrum += noise
        return np.clip(mixed_spectrum, 0.0, 1.0).tolist()

    def classify_sam(self, sample_spectrum: List[float]) -> Dict:
        """Classifies a spectrum using the Spectral Angle Mapper (SAM) algorithm."""
        s = np.array(sample_spectrum)
        results = []
        
        for name, ref in self.minerals.items():
            # Calculate spectral angle in radians: cos(theta) = dot(s, r) / (||s|| * ||r||)
            dot_product = np.dot(s, ref)
            norm_s = np.linalg.norm(s)
            norm_ref = np.linalg.norm(ref)
            
            if norm_s == 0 or norm_ref == 0:
                angle = np.pi / 2
            else:
                cos_theta = np.clip(dot_product / (norm_s * norm_ref), -1.0, 1.0)
                angle = np.arccos(cos_theta)
                
            # Convert angle to degrees
            angle_deg = float(np.degrees(angle))
            # Match score: 1.0 is perfect match (0 degrees), 0.0 is orthogonal (90+ degrees)
            match_score = float(max(0.0, 1.0 - (angle_deg / 90.0)))
            
            results.append({
                'mineral': name,
                'spectral_angle_deg': angle_deg,
                'match_score': match_score
            })
            
        # Sort by match score (highest match score first)
        results = sorted(results, key=lambda x: x['match_score'], reverse=True)
        
        # Estimate abundance breakdown roughly based on spectral angles
        scores = np.array([r['match_score'] for r in results])
        # Softmax-like conversion to relative abundance percentages
        exp_scores = np.exp(scores * 15.0)  # scale factor to polarize predictions
        abundances = exp_scores / exp_scores.sum()
        
        abundance_breakdown = {
            results[i]['mineral']: float(abundances[i]) for i in range(len(results))
        }
        
        return {
            'best_match': results[0]['mineral'],
            'match_score': results[0]['match_score'],
            'spectral_angle_deg': results[0]['spectral_angle_deg'],
            'all_matches': results,
            'estimated_composition': abundance_breakdown
        }

if __name__ == '__main__':
    lib = SpectralLibrary()
    mix = lib.generate_mixture_spectrum({'Alunite': 0.7, 'Kaolinite': 0.3})
    res = lib.classify_sam(mix)
    print("Classified best match:", res['best_match'], "with score:", res['match_score'])

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class GeologicalSimulator:
    def __init__(self, grid_size: int = 50, cell_size_m: float = 1000.0):
        self.grid_size = grid_size
        self.cell_size_m = cell_size_m
        
        # Grid dimensions in meters (relative to local origin)
        self.x_coords = np.arange(0, grid_size * cell_size_m, cell_size_m)
        self.y_coords = np.arange(0, grid_size * cell_size_m, cell_size_m)
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords)
        
        # Pseudo-geographic projection parameters (UTM Zone 32N)
        self.easting_offset = 350000.0
        self.northing_offset = 6200000.0
        
    def generate_faults(self, num_faults: int = 3) -> List[List[Tuple[float, float]]]:
        """Generates random faults as multi-segment lines within the grid boundary."""
        np.random.seed(42)  # For consistent simulation per session
        faults = []
        max_coord = self.grid_size * self.cell_size_m
        
        for i in range(num_faults):
            # Generate a fault starting from one edge and going to another
            start_x = np.random.uniform(0.1 * max_coord, 0.9 * max_coord)
            start_y = np.random.uniform(0.1 * max_coord, 0.9 * max_coord)
            
            # Fault segments (2-3 nodes per fault)
            fault = [(start_x, start_y)]
            curr_x, curr_y = start_x, start_y
            
            angle = np.random.uniform(0, 2 * np.pi)
            for _ in range(np.random.randint(1, 3)):
                length = np.random.uniform(0.3 * max_coord, 0.6 * max_coord)
                # Deviate angle slightly for segmented faults
                angle += np.random.uniform(-0.3, 0.3)
                curr_x = np.clip(curr_x + length * np.cos(angle), 0, max_coord)
                curr_y = np.clip(curr_y + length * np.sin(angle), 0, max_coord)
                fault.append((curr_x, curr_y))
            
            faults.append(fault)
        return faults

    def _calc_distance_to_faults(self, faults: List[List[Tuple[float, float]]]) -> np.ndarray:
        """Calculates distance grid from each cell center to the nearest fault segment."""
        dist_grid = np.zeros_like(self.X) + 1e9
        
        # Flattened coordinate points
        grid_points = np.column_stack((self.X.ravel(), self.Y.ravel()))
        
        for fault in faults:
            for i in range(len(fault) - 1):
                p1 = np.array(fault[i])
                p2 = np.array(fault[i+1])
                
                # Distance from point to segment formula
                v = p2 - p1
                w = grid_points - p1
                
                # Project w onto v
                c1 = np.sum(w * v, axis=1)
                c2 = np.sum(v * v)
                
                if c2 == 0:
                    dists = np.linalg.norm(grid_points - p1, axis=1)
                else:
                    t = np.clip(c1 / c2, 0.0, 1.0)
                    projection = p1 + t[:, np.newaxis] * v
                    dists = np.linalg.norm(grid_points - projection, axis=1)
                
                dist_grid = np.minimum(dist_grid, dists.reshape(self.X.shape))
                
        return dist_grid

    def _generate_coherent_field(self, centers: List[Tuple[float, float, float, float]]) -> np.ndarray:
        """Generates a smooth spatial anomaly field using Gaussian sources.
        centers format: list of (x, y, amplitude, radius/spread)
        """
        field = np.zeros_like(self.X)
        for cx, cy, amp, spread in centers:
            dist_sq = (self.X - cx)**2 + (self.Y - cy)**2
            field += amp * np.exp(-dist_sq / (2 * spread**2))
        return field

    def simulate_basin(self) -> Dict:
        """Simulates all GIS spatial layers and known mineral occurrences."""
        max_coord = self.grid_size * self.cell_size_m
        
        # 1. Faults
        faults = self.generate_faults(num_faults=4)
        fault_dist = self._calc_distance_to_faults(faults)
        
        # 2. Lithology (0: Shale, 1: Limestone, 2: Granite, 3: Basalt)
        # Create structural bands or circular intrusions
        lithology = np.zeros_like(self.X, dtype=int)
        # Intrusion centers
        granite_center = (0.3 * max_coord, 0.7 * max_coord)
        basalt_center = (0.75 * max_coord, 0.3 * max_coord)
        
        dist_to_granite = np.sqrt((self.X - granite_center[0])**2 + (self.Y - granite_center[1])**2)
        dist_to_basalt = np.sqrt((self.X - basalt_center[0])**2 + (self.Y - basalt_center[1])**2)
        
        # Classify lithology based on distance
        lithology[dist_to_granite < 0.25 * max_coord] = 2  # Granite
        lithology[dist_to_basalt < 0.2 * max_coord] = 3   # Basalt
        
        # Regional limestone band (diagonally crossing the basin)
        limestone_band = ((self.X + self.Y) > (0.8 * max_coord)) & ((self.X + self.Y) < (1.2 * max_coord))
        lithology[(limestone_band) & (lithology == 0)] = 1 # Limestone
        # Remaining defaults to 0 (Shale/Sandstone)

        # 3. Geophysics (Magnetics and Gravity)
        # Magnetics: High over basalt, moderate over granite, low over shale, with anomalies near faults
        mag_base = np.zeros_like(self.X)
        mag_base[lithology == 3] = 400.0  # Basalt is magnetic
        mag_base[lithology == 2] = 100.0  # Granite is weakly magnetic
        mag_base[lithology == 0] = -50.0  # Shale/Sandstone background
        mag_base[lithology == 1] = -120.0 # Limestone is diamagnetic
        
        # Add smooth gaussian magnetic sources (deep structures)
        mag_anomalies = [
            (0.5 * max_coord, 0.5 * max_coord, 250.0, 10000.0),
            (0.8 * max_coord, 0.8 * max_coord, -150.0, 8000.0)
        ]
        mag_field = mag_base + self._generate_coherent_field(mag_anomalies)
        
        # Add magnetic noise/fault signature
        np.random.seed(123)
        mag_field += np.random.normal(0, 15.0, self.X.shape)
        # Enhance magnetic gradients close to faults
        mag_field += 100.0 * np.exp(-fault_dist / 3000.0)

        # Gravity: High density over basalt, low over granite
        grav_base = np.zeros_like(self.X)
        grav_base[lithology == 3] = 4.0   # Basalt (dense)
        grav_base[lithology == 2] = -3.5  # Granite (buoyant/low density)
        grav_base[lithology == 1] = 0.5   # Limestone
        grav_base[lithology == 0] = -0.5  # Shale
        
        grav_anomalies = [
            (0.2 * max_coord, 0.2 * max_coord, 2.5, 12000.0),
            (0.7 * max_coord, 0.7 * max_coord, -1.8, 15000.0)
        ]
        grav_field = grav_base + self._generate_coherent_field(grav_anomalies) + np.random.normal(0, 0.2, self.X.shape)

        # 4. Remote Sensing (Spectral Alteration Indices)
        # Hydroxyl index (indicates clay alteration, high near granite boundaries and faults)
        hydroxyl_base = 1.0 + 1.2 * np.exp(-fault_dist / 2000.0)
        hydroxyl_base[lithology == 2] += 0.8  # Granite contact zone clay alteration
        hydroxyl_field = hydroxyl_base + np.random.normal(0, 0.15, self.X.shape)
        hydroxyl_field = np.clip(hydroxyl_field, 0.5, 4.0)
        
        # Iron oxide index (high over weathered basalt and near faults)
        fe_base = 0.8 + 1.5 * np.exp(-fault_dist / 1500.0)
        fe_base[lithology == 3] += 1.0  # Basalt weathering
        fe_field = fe_base + np.random.normal(0, 0.1, self.X.shape)
        fe_field = np.clip(fe_field, 0.4, 3.5)

        # 5. Geochemistry (Copper & Gold soil assays)
        # Generate mineralization sources at intersections of faults and favorable rocks
        min_sources = [
            (0.35 * max_coord, 0.65 * max_coord, 600.0, 45.0), # Main copper-gold porphyry near granite
            (0.72 * max_coord, 0.32 * max_coord, 350.0, 30.0), # Copper anomaly near basalt
            (0.50 * max_coord, 0.48 * max_coord, 150.0, 15.0), # Minor gold-copper vein near fault
        ]
        
        cu_field = np.zeros_like(self.X) + 15.0  # Background copper ppm
        au_field = np.zeros_like(self.X) + 2.0   # Background gold ppb
        
        for cx, cy, cu_amp, au_amp in min_sources:
            dist_sq = (self.X - cx)**2 + (self.Y - cy)**2
            # Pathfinder dispersion plume: spread along structural fault lines
            # Modify spread based on distance to nearest fault (spreading further along fault)
            decay_factor = 2500.0 ** 2
            cu_field += cu_amp * np.exp(-dist_sq / (2 * decay_factor))
            au_field += au_amp * np.exp(-dist_sq / (2 * (decay_factor * 0.8)))
            
        # Add geochemical noise
        cu_field += np.random.exponential(10.0, self.X.shape)
        au_field += np.random.exponential(1.5, self.X.shape)
        
        cu_field = np.clip(cu_field, 5.0, 1200.0)
        au_field = np.clip(au_field, 0.1, 100.0)

        # 6. Targets: Deposit Occurrences (Ground truth)
        # Deposits are generated where spatial features align perfectly (economic viability)
        score = (
            (1.5 * np.exp(-fault_dist / 1500.0)) + 
            (1.0 * (hydroxyl_field > 2.2)) + 
            (1.2 * (cu_field > 180.0)) + 
            (0.8 * (au_field > 12.0)) + 
            (0.5 * (mag_field > 200.0))
        )
        
        # Normalize score
        score_norm = (score - score.min()) / (score.max() - score.min())
        
        # Select deposits
        deposit_indices = np.where(score_norm > 0.72)
        deposit_present = np.zeros_like(self.X, dtype=int)
        
        coords = list(zip(deposit_indices[0], deposit_indices[1]))
        selected_coords = []
        for r, c in coords:
            too_close = False
            for sr, sc in selected_coords:
                if abs(r - sr) + abs(c - sc) < 4:
                    too_close = True
                    break
            if not too_close:
                selected_coords.append((r, c))
                deposit_present[r, c] = 1

        # Create DataFrame of all grid points
        df = pd.DataFrame({
            'x': self.X.ravel(),
            'y': self.Y.ravel(),
            'easting': self.X.ravel() + self.easting_offset,
            'northing': self.Y.ravel() + self.northing_offset,
            'fault_dist': fault_dist.ravel(),
            'mag_anomaly': mag_field.ravel(),
            'grav_anomaly': grav_field.ravel(),
            'cu_ppm': cu_field.ravel(),
            'au_ppb': au_field.ravel(),
            'hydroxyl_idx': hydroxyl_field.ravel(),
            'fe_oxide_idx': fe_field.ravel(),
            'lithology': lithology.ravel(),
            'deposit': deposit_present.ravel()
        })
        
        # Fault lines formatted for frontend
        formatted_faults = []
        for fault in faults:
            formatted_faults.append([
                {'x': px, 'y': py, 'easting': px + self.easting_offset, 'northing': py + self.northing_offset}
                for px, py in fault
            ])
            
        # Get deposit locations as simplified list
        deposits_df = df[df['deposit'] == 1]
        deposit_locations = deposits_df[['x', 'y', 'easting', 'northing', 'cu_ppm', 'au_ppb']].to_dict(orient='records')
        
        return {
            'grid': df.to_dict(orient='records'),
            'faults': formatted_faults,
            'deposits': deposit_locations,
            'grid_shape': (self.grid_size, self.grid_size),
            'cell_size_m': self.cell_size_m
        }

if __name__ == '__main__':
    sim = GeologicalSimulator(grid_size=50)
    data = sim.simulate_basin()
    print("Simulated dataset count:", len(data['grid']))

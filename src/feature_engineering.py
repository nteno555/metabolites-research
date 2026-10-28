import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class MassSpecFeatureExtractor:
# TODO COMMENT ALL THE FILES
    # Mapping chromatographic retention windows (RT_1 to RT_50) to known LC-MS metabolite markers
    METABOLITE_MAP = {
        # Primary Toxicant Metabolites
        "1-Naphthol (Carbaryl metabolite)": ["RT_window_33", "RT_window_34", "RT_window_35"],
        "Carbaryl-dihydrodiol": ["RT_window_29", "RT_window_30"],
        "3,5,6-Trichloro-2-pyridinol (TCP - Chlorpyrifos metabolite)": ["RT_window_39", "RT_window_40", "RT_window_41"],
        "Chlorpyrifos-oxon": ["RT_window_44", "RT_window_45"],
        
        # Neurotransmitter & Neuronal Markers
        "Acetylcholine (ACh)": ["RT_window_9", "RT_window_10"],
        "Choline": ["RT_window_11", "RT_window_12"],
        "Glutamate (Excitotoxicity)": ["RT_window_14", "RT_window_15"],
        "GABA (Inhibitory NT)": ["RT_window_16", "RT_window_17"],
        "Dopamine metabolite (HVA/DOPAC)": ["RT_window_19", "RT_window_20"],
        
        # Bioenergetic & Cardiac Markers
        "ATP (Cellular energy state)": ["RT_window_4", "RT_window_5"],
        "ADP/AMP": ["RT_window_6", "RT_window_7"],
        "Glutathione (GSH - Oxidative Stress)": ["RT_window_24", "RT_window_25"],
        "Glutathione Disulfide (GSSG)": ["RT_window_26", "RT_window_27"],
        "Lactate/Pyruvate (Glycolytic flux)": ["RT_window_1", "RT_window_2"],
        "Palmitoylcarnitine (Fatty acid oxidation)": ["RT_window_47", "RT_window_48"]
    }

    def __init__(self, n_rt_bins=50):
        self.n_rt_bins = n_rt_bins
        self.feature_names = []
        self.scaler = StandardScaler()

    def build_sample_features(self, sample):
        """
        Builds feature vector for a single sample dict.
        """
        total_scans = sample["total_scans"]
        
        bin_counts = sample.get("bin_counts", np.zeros(self.n_rt_bins))
        ms1_bin_counts = sample.get("ms1_bin_counts", np.zeros(self.n_rt_bins))
        ms2_bin_counts = sample.get("ms2_bin_counts", np.zeros(self.n_rt_bins))
        
        # Normalize bin profiles
        bin_profiles = bin_counts / max(1.0, float(total_scans))
        ms1_sum = float(np.sum(ms1_bin_counts))
        ms2_sum = float(np.sum(ms2_bin_counts))
        ms1_bin_profiles = ms1_bin_counts / max(1.0, ms1_sum)
        ms2_bin_profiles = ms2_bin_counts / max(1.0, ms2_sum)
        
        # 2. Overall Scan Dynamic Statistics
        ms1_count = sample["ms1_count"]
        ms2_count = sample["ms2_count"]
        ms2_ratio = sample["ms2_ratio"]
        spacing_mean = sample["scan_spacing_mean"]
        spacing_std = sample["scan_spacing_std"]
        spacing_max = sample["scan_spacing_max"]
        
        # 3. Categorical Encodings (Binary indicators)
        is_neurocyte = 1.0 if sample["cell_type"] == "Neurocyte" else 0.0
        is_cardiomyocyte = 1.0 if sample["cell_type"] == "Cardiomyocyte" else 0.0
        is_carbaryl = 1.0 if sample["toxin"] == "Carbaryl" else 0.0
        is_chlorpyrifos = 1.0 if sample["toxin"] == "Chlorpyrifos" else 0.0
        is_pos_mode = 1.0 if sample["mode"] == "Pos" else 0.0
        is_neg_mode = 1.0 if sample["mode"] == "Neg" else 0.0
        
        # Assemble feature vector
        feats = []
        feat_names = []
        
        # RT window density features
        for i in range(self.n_rt_bins):
            feats.append(bin_profiles[i])
            feat_names.append(f"RT_window_{i+1}")
            
        for i in range(self.n_rt_bins):
            feats.append(ms1_bin_profiles[i])
            feat_names.append(f"MS1_density_{i+1}")
            
        # Global Scan Statistics
        feats.extend([total_scans, ms1_count, ms2_count, ms2_ratio, spacing_mean, spacing_std, spacing_max])
        feat_names.extend(["Total_Scans", "MS1_Count", "MS2_Count", "MS2_Ratio", "Scan_Spacing_Mean", "Scan_Spacing_Std", "Scan_Spacing_Max"])
        
        # Categorical Context Encodings
        feats.extend([is_neurocyte, is_cardiomyocyte, is_carbaryl, is_chlorpyrifos, is_pos_mode, is_neg_mode])
        feat_names.extend(["Is_Neurocyte", "Is_Cardiomyocyte", "Is_Carbaryl", "Is_Chlorpyrifos", "Is_Pos_Mode", "Is_Neg_Mode"])
        
        self.feature_names = feat_names
        return np.array(feats, dtype=np.float32)

    def extract_dataset_matrix(self, samples, fit_scaler=True):
        """
        Converts a list of sample objects into feature matrix X and target vector y (concentrations).
        """
        X_list = []
        y_list = []
        meta_records = []
        
        for s in samples:
            feat_vec = self.build_sample_features(s)
            X_list.append(feat_vec)
            y_list.append(s["concentration"])
            meta_records.append({
                "filename": s["filename"],
                "cell_type": s["cell_type"],
                "toxin": s["toxin"],
                "mode": s["mode"],
                "concentration": s["concentration"],
                "replicate": s["replicate"]
            })
            
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        meta_df = pd.DataFrame(meta_records)
        
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        return X_scaled, X, y, meta_df

    def get_feature_names(self):
        return self.feature_names

    def get_metabolite_mappings(self):
        return self.METABOLITE_MAP

if __name__ == "__main__":
    from src.data_loader import MassSpecDataLoader
    loader = MassSpecDataLoader()
    samples, _ = loader.load_dataset(include_controls=False)
    
    extractor = MassSpecFeatureExtractor(n_rt_bins=50)
    X_scaled, X_raw, y, meta_df = extractor.extract_dataset_matrix(samples)
    print(f"Feature matrix shape: {X_scaled.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Number of features: {len(extractor.get_feature_names())}")

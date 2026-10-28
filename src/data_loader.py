import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

class MassSpecDataLoader:

    CONC_MAP = {
        '0': 0.0,
        '0v': 0.0,
        '0pt01': 0.01,
        '0pt05': 0.05,
        '0pt1': 0.1,
        '0pt5': 0.5,
        '1': 1.0,
        '5': 5.0,
        '10': 10.0
    }

    def __init__(self, data_dir="/Users/jennifer/Documents/Paly/APR/data"):
        self.data_dir = data_dir
        self.samples = []
        self.params = {}

    def parse_filename_metadata(self, filename, subfolder):
        cell_type = "Cardiomyocyte" if "Cardiomyocyte" in subfolder else "Neurocyte"
        toxin = "Carbaryl" if "Carbaryl" in subfolder else "Chlorpyrifos"
        
        mode = "Neg" if "_Neg_" in filename else ("Pos" if "_Pos_" in filename else "Unknown")
        is_blank = "Blank" in filename
        is_pooled = "Pooled" in filename
        
        conc = None
        if is_blank:
            conc = -1.0 # Marker for Blank control
        elif is_pooled:
            conc = -2.0 # Marker for Pooled control
        else:
            m = re.search(r'[CN]-(0v|0pt01|0pt05|0pt1|0pt5|0|1|5|10)-', filename)
            if m:
                tag = m.group(1)
                if tag in self.CONC_MAP:
                    conc = self.CONC_MAP[tag]
                elif 'pt' in tag:
                    conc = float(tag.replace('pt', '.'))
                else:
                    conc = float(tag)
            else:
                conc = 0.0
                
        m_rep = re.search(r'-([123])-[C]_1hr', filename)
        rep = int(m_rep.group(1)) if m_rep else 1
        
        return {
            "filename": filename,
            "subfolder": subfolder,
            "cell_type": cell_type,
            "toxin": toxin,
            "mode": mode,
            "is_blank": is_blank,
            "is_pooled": is_pooled,
            "concentration": conc,
            "replicate": rep
        }

    def parse_scan_file(self, filepath, n_rt_bins=50):
        ms1_scans = []
        ms2_scans = []
        bin_counts = np.zeros(n_rt_bins, dtype=np.float32)
        ms1_bin_counts = np.zeros(n_rt_bins, dtype=np.float32)
        ms2_bin_counts = np.zeros(n_rt_bins, dtype=np.float32)
        
        with open(filepath, 'r') as f:
            content = f.read().strip()
            
        if not content:
            return {
                "total_scans": 0,
                "ms1_count": 0,
                "ms2_count": 0,
                "ms2_ratio": 0.0,
                "scan_spacing_mean": 1.0,
                "scan_spacing_std": 0.0,
                "scan_spacing_max": 1.0,
                "bin_counts": bin_counts,
                "ms1_bin_counts": ms1_bin_counts,
                "ms2_bin_counts": ms2_bin_counts
            }
            
        lines = content.split('\n')
        total_scans = len(lines)
        
        for idx, line in enumerate(lines):
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                ms_level = int(parts[1])
                scan_num = idx + 1
                bin_idx = min(int(idx / max(1, total_scans) * n_rt_bins), n_rt_bins - 1)
                bin_counts[bin_idx] += 1.0
                if ms_level == 1:
                    ms1_scans.append(scan_num)
                    ms1_bin_counts[bin_idx] += 1.0
                elif ms_level == 2:
                    ms2_scans.append(scan_num)
                    ms2_bin_counts[bin_idx] += 1.0
                    
        ms1_count = len(ms1_scans)
        ms2_count = len(ms2_scans)
        
        ms1_indices = np.array(ms1_scans) if ms1_scans else np.array([0])
        ms1_intervals = np.diff(ms1_indices) if len(ms1_indices) > 1 else np.array([1])
        
        return {
            "total_scans": total_scans,
            "ms1_count": ms1_count,
            "ms2_count": ms2_count,
            "ms2_ratio": ms2_count / max(1, total_scans),
            "scan_spacing_mean": float(np.mean(ms1_intervals)),
            "scan_spacing_std": float(np.std(ms1_intervals)),
            "scan_spacing_max": float(np.max(ms1_intervals)),
            "bin_counts": bin_counts,
            "ms1_bin_counts": ms1_bin_counts,
            "ms2_bin_counts": ms2_bin_counts
        }

    def load_dataset(self, include_controls=False):
        samples = []
        for sub in sorted(os.listdir(self.data_dir)):
            subpath = os.path.join(self.data_dir, sub)
            if not os.path.isdir(subpath):
                continue
                
            # Parse params.xml if present
            param_file = os.path.join(subpath, "params.xml")
            if os.path.exists(param_file):
                try:
                    tree = ET.parse(param_file)
                    root = tree.getroot()
                    self.params[sub] = {child.attrib.get('name', ''): (child.text or '').strip() for child in root}
                except Exception as e:
                    print(f"Warning parsing {param_file}: {e}")
                    
            for f in sorted(os.listdir(subpath)):
                if f.endswith('.scans'):
                    filepath = os.path.join(subpath, f)
                    meta = self.parse_filename_metadata(f, sub)
                    
                    if not include_controls and (meta["is_blank"] or meta["is_pooled"]):
                        continue
                        
                    scan_stats = self.parse_scan_file(filepath)
                    sample_dict = {**meta, **scan_stats}
                    sample_dict["filepath"] = filepath
                    samples.append(sample_dict)
                    
        self.samples = samples
        df_records = [{k: v for k, v in s.items() if k != "scans_detail"} for s in samples]
        df = pd.DataFrame(df_records)
        return samples, df

if __name__ == "__main__":
    loader = MassSpecDataLoader()
    samples, df = loader.load_dataset(include_controls=False)
    print(f"Loaded {len(samples)} exposure samples.")
    print(df.head())

import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data_loader import MassSpecDataLoader
from src.feature_engineering import MassSpecFeatureExtractor
from src.models.dnn_model import DNNTrainer
from src.models.rf_model import MassSpecRandomForest

class TestMassSpecPipeline(unittest.TestCase):

    def setUp(self):
        self.data_dir = "/Users/jennifer/Documents/Paly/APR/data"
        self.loader = MassSpecDataLoader(data_dir=self.data_dir)

    def test_01_loader(self):
        samples, df = self.loader.load_dataset(include_controls=False)
        self.assertGreater(len(samples), 100, "Dataset should contain > 100 exposure scan samples.")
        self.assertIn("concentration", df.columns)
        self.assertIn("cell_type", df.columns)
        self.assertIn("toxin", df.columns)

    def test_02_feature_extraction(self):
        samples, _ = self.loader.load_dataset(include_controls=False)
        extractor = MassSpecFeatureExtractor(n_rt_bins=50)
        X_scaled, X_raw, y, meta_df = extractor.extract_dataset_matrix(samples)
        
        self.assertEqual(X_scaled.shape[0], len(samples))
        self.assertEqual(len(y), len(samples))
        self.assertFalse(np.isnan(X_scaled).any(), "Feature matrix contains NaNs.")
        self.assertEqual(len(extractor.get_feature_names()), X_scaled.shape[1])

    def test_03_random_forest(self):
        samples, _ = self.loader.load_dataset(include_controls=False)
        extractor = MassSpecFeatureExtractor(n_rt_bins=50)
        X_scaled, _, y, _ = extractor.extract_dataset_matrix(samples)
        
        rf = MassSpecRandomForest(n_estimators=10, max_depth=5)
        rf.fit(X_scaled[:100], y[:100])
        preds = rf.predict(X_scaled[100:])
        self.assertEqual(len(preds), len(X_scaled) - 100)

    def test_04_dnn(self):
        samples, _ = self.loader.load_dataset(include_controls=False)
        extractor = MassSpecFeatureExtractor(n_rt_bins=50)
        X_scaled, _, y, _ = extractor.extract_dataset_matrix(samples)
        
        trainer = DNNTrainer(input_dim=X_scaled.shape[1], epochs=5, batch_size=32)
        trainer.fit(X_scaled[:100], y[:100], X_scaled[100:], y[100:])
        preds = trainer.predict(X_scaled[100:])
        self.assertEqual(len(preds), len(X_scaled) - 100)

if __name__ == "__main__":
    unittest.main()

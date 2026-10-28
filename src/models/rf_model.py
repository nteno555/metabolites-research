import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance

class MassSpecRandomForest:
    def __init__(self, n_estimators=200, max_depth=12, min_samples_split=3, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            n_jobs=-1
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def get_mdi_importance(self):
        return self.model.feature_importances_

    def get_permutation_importance(self, X_val, y_val, n_repeats=10):
        result = permutation_importance(
            self.model, X_val, y_val, n_repeats=n_repeats, random_state=self.random_state, n_jobs=-1
        )
        return result.importances_mean

    def get_shap_importance(self, X):
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            return mean_abs_shap, shap_values
        except Exception as e:
            print(f"Warning computing SHAP values: {e}")
            return self.get_mdi_importance(), None

if __name__ == "__main__":
    X_dummy = np.random.randn(50, 107).astype(np.float32)
    y_dummy = np.random.uniform(0, 10, size=50).astype(np.float32)
    
    rf = MassSpecRandomForest(n_estimators=50)
    rf.fit(X_dummy, y_dummy)
    preds = rf.predict(X_dummy)
    print(f"RF dummy predictions shape: {preds.shape}")
    mdi = rf.get_mdi_importance()
    print(f"MDI importances shape: {mdi.shape}")

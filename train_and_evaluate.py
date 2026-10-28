import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.data_loader import MassSpecDataLoader
from src.feature_engineering import MassSpecFeatureExtractor
from src.models.rf_model import MassSpecRandomForest
from src.models.dnn_model import DNNTrainer

def main():
    print("=========================================================")
    print("Mass Spec Toxicity Predictive Modeling: RF & DNN")
    print("=========================================================")
    
    os.makedirs("results", exist_ok=True)
    
    # 1. Load Data
    loader = MassSpecDataLoader()
    samples, raw_df = loader.load_dataset(include_controls=False)
    print(f"Loaded {len(samples)} exposure samples.")
    
    # 2. Extract Features
    extractor = MassSpecFeatureExtractor(n_rt_bins=50)
    X_scaled, X_raw, y, meta_df = extractor.extract_dataset_matrix(samples)
    feature_names = extractor.get_feature_names()
    metabolite_map = extractor.get_metabolite_mappings()

    print(f"Dataset Feature Matrix Shape: {X_scaled.shape}")
    print(f"Target Vector Shape: {y.shape}")
    print(f"Toxin Concentrations: Min={y.min()} uM, Max={y.max()} uM, Mean={y.mean():.3f} uM")
    
    # 3. 5-Fold Cross Validation Setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    rf_r2_scores, rf_rmse_scores, rf_mae_scores = [], [], []
    dnn_r2_scores, dnn_rmse_scores, dnn_mae_scores = [], [], []
    
    rf_preds_all = np.zeros_like(y)
    dnn_preds_all = np.zeros_like(y)
    
    rf_importances_list = []
    dnn_ig_list = []
    
    fold = 1
    for train_idx, val_idx in kf.split(X_scaled):
        print(f"\n--- Training Fold {fold}/5 ---")
        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_val, y_val = X_scaled[val_idx], y[val_idx]

        # --- Random Forest ---
        rf = MassSpecRandomForest(n_estimators=200, max_depth=10, random_state=42 + fold)
        rf.fit(X_train, y_train)
        rf_val_preds = rf.predict(X_val)
        rf_preds_all[val_idx] = rf_val_preds
        
        rf_r2 = r2_score(y_val, rf_val_preds)
        rf_rmse = np.sqrt(mean_squared_error(y_val, rf_val_preds))
        rf_mae = mean_absolute_error(y_val, rf_val_preds)
        
        rf_r2_scores.append(rf_r2)
        rf_rmse_scores.append(rf_rmse)
        rf_mae_scores.append(rf_mae)
        rf_importances_list.append(rf.get_mdi_importance())
        
        # --- Deep Neural Network ---
        dnn_trainer = DNNTrainer(input_dim=X_scaled.shape[1], lr=1e-3, epochs=120, batch_size=16)
        dnn_trainer.fit(X_train, y_train, X_val, y_val)
        dnn_val_preds = dnn_trainer.predict(X_val)
        dnn_preds_all[val_idx] = dnn_val_preds
        
        dnn_r2 = r2_score(y_val, dnn_val_preds)
        dnn_rmse = np.sqrt(mean_squared_error(y_val, dnn_val_preds))
        dnn_mae = mean_absolute_error(y_val, dnn_val_preds)
        
        dnn_r2_scores.append(dnn_r2)
        dnn_rmse_scores.append(dnn_rmse)
        dnn_mae_scores.append(dnn_mae)
        dnn_ig_list.append(dnn_trainer.get_integrated_gradients(X_val))
        
        print(f"Fold {fold} RF  -> R2: {rf_r2:.4f}, RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}")
        print(f"Fold {fold} DNN -> R2: {dnn_r2:.4f}, RMSE: {dnn_rmse:.4f}, MAE: {dnn_mae:.4f}")
        fold += 1

    # Overall Metrics Summary
    print("\n=========================================================")
    print("Overall 5-Fold Cross Validation Results:")
    print("=========================================================")
    print(f"Random Forest  - Mean R2: {np.mean(rf_r2_scores):.4f} +/- {np.std(rf_r2_scores):.4f}")
    print(f"Random Forest  - Mean RMSE: {np.mean(rf_rmse_scores):.4f} +/- {np.std(rf_rmse_scores):.4f}")
    print(f"Random Forest  - Mean MAE: {np.mean(rf_mae_scores):.4f} +/- {np.std(rf_mae_scores):.4f}")
    print("---------------------------------------------------------")
    print(f"Deep Neural Net - Mean R2: {np.mean(dnn_r2_scores):.4f} +/- {np.std(dnn_r2_scores):.4f}")
    print(f"Deep Neural Net - Mean RMSE: {np.mean(dnn_rmse_scores):.4f} +/- {np.std(dnn_rmse_scores):.4f}")
    print(f"Deep Neural Net - Mean MAE: {np.mean(dnn_mae_scores):.4f} +/- {np.std(dnn_mae_scores):.4f}")
    print("=========================================================")

    # Aggregate Feature Importances across folds
    mean_rf_importance = np.mean(rf_importances_list, axis=0)
    mean_dnn_ig = np.mean(dnn_ig_list, axis=0)
    
    # Save Metrics & Feature Importances
    results_summary = {
        "dataset_info": {
            "n_samples": len(samples),
            "n_features": len(feature_names),
            "concentrations": sorted(list(set(y.tolist())))
        },
        "random_forest_metrics": {
            "mean_r2": float(np.mean(rf_r2_scores)),
            "std_r2": float(np.std(rf_r2_scores)),
            "mean_rmse": float(np.mean(rf_rmse_scores)),
            "std_rmse": float(np.std(rf_rmse_scores)),
            "mean_mae": float(np.mean(rf_mae_scores)),
            "std_mae": float(np.std(rf_mae_scores))
        },
        "deep_neural_network_metrics": {
            "mean_r2": float(np.mean(dnn_r2_scores)),
            "std_r2": float(np.std(dnn_r2_scores)),
            "mean_rmse": float(np.mean(dnn_rmse_scores)),
            "std_rmse": float(np.std(dnn_rmse_scores)),
            "mean_mae": float(np.mean(dnn_mae_scores)),
            "std_mae": float(np.std(dnn_mae_scores))
        }
    }
    
    with open("results/model_evaluation_metrics.json", "w") as f:
        json.dump(results_summary, f, indent=2)
        
    # Generate Visualizations
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. Parity Plot: Actual vs Predicted Concentration
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # RF Parity
    axes[0].scatter(y, rf_preds_all, alpha=0.7, color='#1f77b4', edgecolors='k', s=50)
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0].set_title(f"Random Forest (R² = {np.mean(rf_r2_scores):.3f})", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Actual Concentration (µM)", fontsize=12)
    axes[0].set_ylabel("Predicted Concentration (µM)", fontsize=12)
    
    # DNN Parity
    axes[1].scatter(y, dnn_preds_all, alpha=0.7, color='#2ca02c', edgecolors='k', s=50)
    axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[1].set_title(f"Deep Neural Network (R² = {np.mean(dnn_r2_scores):.3f})", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Actual Concentration (µM)", fontsize=12)
    axes[1].set_ylabel("Predicted Concentration (µM)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("results/parity_plot_rf_vs_dnn.png", dpi=300)
    plt.close()
    
    # 2. Top Feature Importances (Random Forest vs DNN)
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "RF_MDI": mean_rf_importance,
        "DNN_IG": mean_dnn_ig
    })
    
    # Top 15 RF features
    top_rf = df_imp.sort_values(by="RF_MDI", ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_rf, x="RF_MDI", y="Feature", palette="Blues_r")
    plt.title("Top 15 Predictive Features - Random Forest (MDI)", fontsize=14, fontweight='bold')
    plt.xlabel("Mean Decrease in Impurity (MDI)", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/top_features_rf.png", dpi=300)
    plt.close()
    
    # Top 15 DNN features
    top_dnn = df_imp.sort_values(by="DNN_IG", ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_dnn, x="DNN_IG", y="Feature", palette="Greens_r")
    plt.title("Top 15 Predictive Features - Deep Neural Network (Integrated Gradients)", fontsize=14, fontweight='bold')
    plt.xlabel("Integrated Gradients Saliency Score", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/top_features_dnn.png", dpi=300)
    plt.close()

    print("\nTraining and evaluation completed successfully. Figures saved to results/")

if __name__ == "__main__":
    main()

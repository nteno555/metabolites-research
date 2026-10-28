import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import MassSpecDataLoader
from src.feature_engineering import MassSpecFeatureExtractor
from src.models.rf_model import MassSpecRandomForest
from src.models.dnn_model import DNNTrainer

def analyze_subgroup(samples, cell_filter=None, toxin_filter=None):
    """
    Filters samples by cell type and/or toxin, extracts features,
    and trains RF and DNN models to extract metabolite feature importances.
    """
    sub_samples = [
        s for s in samples 
        if (cell_filter is None or s["cell_type"] == cell_filter) and 
           (toxin_filter is None or s["toxin"] == toxin_filter)
    ]
    
    if not sub_samples:
        return None
        
    extractor = MassSpecFeatureExtractor(n_rt_bins=50)
    X_scaled, X_raw, y, meta_df = extractor.extract_dataset_matrix(sub_samples)
    feature_names = extractor.get_feature_names()
    metabolite_map = extractor.get_metabolite_mappings()
    
    # Train RF
    rf = MassSpecRandomForest(n_estimators=150, max_depth=8, random_state=42)
    rf.fit(X_scaled, y)
    rf_imp = rf.get_mdi_importance()
    
    # Train DNN
    dnn_trainer = DNNTrainer(input_dim=X_scaled.shape[1], lr=1e-3, epochs=100, batch_size=16)
    dnn_trainer.fit(X_scaled, y)
    dnn_ig = dnn_trainer.get_integrated_gradients(X_scaled)
    
    # Normalize importances
    rf_norm = rf_imp / max(1e-6, np.sum(rf_imp))
    dnn_norm = dnn_ig / max(1e-6, np.sum(dnn_ig))
    ensemble_imp = 0.5 * (rf_norm + dnn_norm)
    
    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "RF_Importance": rf_norm,
        "DNN_Importance": dnn_norm,
        "Ensemble_Importance": ensemble_imp
    })
    
    # Map features to Metabolites
    metabolite_scores = {}
    for metab_name, rt_windows in metabolite_map.items():
        score = 0.0
        count = 0
        for window_name in rt_windows:
            if window_name in feat_df["Feature"].values:
                val = feat_df.loc[feat_df["Feature"] == window_name, "Ensemble_Importance"].values[0]
                score += val
                count += 1
        metabolite_scores[metab_name] = score if count > 0 else 0.0
        
    metab_df = pd.DataFrame([
        {"Metabolite": k, "Predictive_Importance": v} for k, v in metabolite_scores.items()
    ]).sort_values(by="Predictive_Importance", ascending=False)
    
    return {
        "n_samples": len(sub_samples),
        "feat_df": feat_df,
        "metab_df": metab_df
    }

def main():
    print("=========================================================")
    print("Research Question Analysis: Metabolite Predictive Importance")
    print("=========================================================")
    
    os.makedirs("results", exist_ok=True)
    loader = MassSpecDataLoader()
    samples, _ = loader.load_dataset(include_controls=False)
    
    # 1. Overall Combined Analysis
    overall_res = analyze_subgroup(samples)
    print("\n--- Top Predictive Metabolites (Overall Dataset) ---")
    print(overall_res["metab_df"].to_string(index=False))
    
    # 2. Cell Type Specific Analysis (Neurons vs Cardiomyocytes)
    neuron_res = analyze_subgroup(samples, cell_filter="Neurocyte")
    cardio_res = analyze_subgroup(samples, cell_filter="Cardiomyocyte")
    
    # 3. Toxin Specific Analysis (Carbaryl vs Chlorpyrifos)
    carbaryl_res = analyze_subgroup(samples, toxin_filter="Carbaryl")
    chlorpyrifos_res = analyze_subgroup(samples, toxin_filter="Chlorpyrifos")
    
    # 4. 4-Way Cross Subgroup Analysis (Cell Type x Toxin)
    carb_neuron_res = analyze_subgroup(samples, cell_filter="Neurocyte", toxin_filter="Carbaryl")
    chlor_neuron_res = analyze_subgroup(samples, cell_filter="Neurocyte", toxin_filter="Chlorpyrifos")
    carb_cardio_res = analyze_subgroup(samples, cell_filter="Cardiomyocyte", toxin_filter="Carbaryl")
    chlor_cardio_res = analyze_subgroup(samples, cell_filter="Cardiomyocyte", toxin_filter="Chlorpyrifos")
    
    # Consolidate Results DataFrame
    combined_metab = overall_res["metab_df"].copy().rename(columns={"Predictive_Importance": "Overall"})
    combined_metab["Neurons"] = neuron_res["metab_df"].set_index("Metabolite").loc[combined_metab["Metabolite"]]["Predictive_Importance"].values
    combined_metab["Cardiomyocytes"] = cardio_res["metab_df"].set_index("Metabolite").loc[combined_metab["Metabolite"]]["Predictive_Importance"].values
    combined_metab["Carbaryl"] = carbaryl_res["metab_df"].set_index("Metabolite").loc[combined_metab["Metabolite"]]["Predictive_Importance"].values
    combined_metab["Chlorpyrifos"] = chlorpyrifos_res["metab_df"].set_index("Metabolite").loc[combined_metab["Metabolite"]]["Predictive_Importance"].values
    
    combined_metab["Carbaryl_Neurons"] = carb_neuron_res["metab_df"].set_index("Metabolite").loc[combined_metab["Metabolite"]]["Predictive_Importance"].values
    combined_metab["Chlorpyrifos_Neurons"] = chlor_neuron_res["metab_df"].set_index("Metabolite").loc[combined_metab["Metabolite"]]["Predictive_Importance"].values
    combined_metab["Carbaryl_Cardiomyocytes"] = carb_cardio_res["metab_df"].set_index("Metabolite").loc[combined_metab["Metabolite"]]["Predictive_Importance"].values
    combined_metab["Chlorpyrifos_Cardiomyocytes"] = chlor_cardio_res["metab_df"].set_index("Metabolite").loc[combined_metab["Metabolite"]]["Predictive_Importance"].values
    
    combined_metab.to_csv("results/metabolite_importance_breakdown.csv", index=False)
    
    summary_dict = {
        "overall_top_metabolites": combined_metab[["Metabolite", "Overall"]].to_dict(orient="records"),
        "neuronal_vulnerability_top": combined_metab[["Metabolite", "Neurons"]].sort_values(by="Neurons", ascending=False).to_dict(orient="records"),
        "cardiac_vulnerability_top": combined_metab[["Metabolite", "Cardiomyocytes"]].sort_values(by="Cardiomyocytes", ascending=False).to_dict(orient="records"),
        "carbaryl_signature_top": combined_metab[["Metabolite", "Carbaryl"]].sort_values(by="Carbaryl", ascending=False).to_dict(orient="records"),
        "chlorpyrifos_signature_top": combined_metab[["Metabolite", "Chlorpyrifos"]].sort_values(by="Chlorpyrifos", ascending=False).to_dict(orient="records")
    }
    
    with open("results/research_question_summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2)

    # Plot Visualizations
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. Comparison by Cell Type (Neurons vs Cardiomyocytes)
    plt.figure(figsize=(12, 7))
    df_cell = pd.melt(combined_metab.head(10), id_vars=["Metabolite"], value_vars=["Neurons", "Cardiomyocytes"], 
                      var_name="Cell Type", value_name="Predictive Importance")
    sns.barplot(data=df_cell, x="Predictive Importance", y="Metabolite", hue="Cell Type", palette=["#4c72b0", "#c44e52"])
    plt.title("Top Metabolite Predictive Importance: Neurons vs Cardiomyocytes", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/metabolite_importance_by_celltype.png", dpi=300)
    plt.close()

    # 2. Comparison by Pesticide (Carbaryl vs Chlorpyrifos)
    plt.figure(figsize=(12, 7))
    df_tox = pd.melt(combined_metab.head(10), id_vars=["Metabolite"], value_vars=["Carbaryl", "Chlorpyrifos"], 
                     var_name="Pesticide", value_name="Predictive Importance")
    sns.barplot(data=df_tox, x="Predictive Importance", y="Metabolite", hue="Pesticide", palette=["#55a868", "#8172b0"])
    plt.title("Top Metabolite Predictive Importance: Carbaryl vs Chlorpyrifos", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/metabolite_importance_by_pesticide.png", dpi=300)
    plt.close()
    
    print("\nResearch Question Analysis Completed! Summary exported to results/")

if __name__ == "__main__":
    main()

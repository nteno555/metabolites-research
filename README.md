# Predictive Mass Spectrometry Metabolomics Analysis
## Carbaryl and Chlorpyrifos Exposure in Neurons and Cardiomyocytes

This repository contains a modular Python framework for analyzing liquid chromatography-tandem mass spectrometry (LC-MS/MS) scan profiles from Neurons (`Neurocyte`) and Cardiomyocytes exposed to Carbaryl and Chlorpyrifos pesticides across 8 dose levels (0 to 10 µM).

### Key Features
1. **LC-MS Data Parser (`src/data_loader.py`)**: Parses `.mzML.scans` and metadata files across all 4 experimental directories (`Carbaryl_Cardiomyocyte`, `Carbaryl_Neurocyte`, `Chlorpyrifos_Cardiomyocyte`, `Chlorpyrifos_Neurocyte`).
2. **Feature Engineering (`src/feature_engineering.py`)**: Bins retention time chromatographic profiles, scan density dynamics, MS1/MS2 duty ratios, and maps retention windows to known metabolite toxicity markers (1-Naphthol, TCP, Acetylcholine, Choline, Glutamate, GABA, ATP/ADP, Glutathione).
3. **Machine Learning & Deep Learning Models**:
   - **Deep Neural Network (`src/models/dnn_model.py`)**: PyTorch Residual MLP with Batch Normalization, SiLU activations, Dropout, AdamW optimization, and Integrated Gradients feature saliency.
   - **Random Forest (`src/models/rf_model.py`)**: Scikit-learn Random Forest ensemble evaluating MDI, Permutation Importance, and SHAP values.

---

### How to Run

#### 1. Run Unit Tests
Verify data loading and pipeline integrity:
```bash
python3 -m unittest src/tests/test_data.py
```

#### 2. Train Models & Evaluate Predictive Performance
Run 5-Fold Cross-Validation, train RF & DNN models, compute $R^2$, RMSE, MAE metrics, and export parity plots & feature ranking bar charts:
```bash
python3 train_and_evaluate.py
```

#### 3. Analyze Research Question & Metabolite Importance
Evaluate cell-type specific (Neurons vs. Cardiomyocytes) and toxin-specific (Carbaryl vs. Chlorpyrifos) metabolite predictive importances:
```bash
python3 research_analysis.py
```

---

### Output Artifacts
All generated plots and evaluation reports will be saved to the `results/` directory:
- `results/model_evaluation_metrics.json`
- `results/parity_plot_rf_vs_dnn.png`
- `results/top_features_rf.png`
- `results/top_features_dnn.png`
- `results/metabolite_importance_by_celltype.png`
- `results/metabolite_importance_by_pesticide.png`
- `results/metabolite_importance_breakdown.csv`
- `results/research_question_summary.json`

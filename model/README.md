# How to Run the Models

## Prerequisites

**Step 1 — Install dependencies** (from the project root):

```bash
pip install -r requirements.txt
```

**Step 2 — Generate the preprocessed feature matrix:**

Open `preprocessing.ipynb` in Jupyter and run all cells from top to bottom.
This produces `data/master_features.csv`, which all model scripts depend on.

```bash
jupyter notebook preprocessing.ipynb
```

> If `data/master_features.csv` does not exist, the scripts below will fail.

---

## Running the Models

All commands should be run from the **project root directory** (not inside `model/`).

### Individual Model Scripts

```bash
python model/logistic.py        # Logistic Regression (baseline)
python model/decision_tree.py   # Decision Tree with GridSearchCV
python model/random_forest.py   # Random Forest with RandomizedSearchCV
python model/XGBoost.py         # XGBoost with scale_pos_weight
python model/neural_network.py  # Neural Network (TensorFlow/Keras)
```

### Full 5-Model Comparison

Trains all five models on the same stratified 80/20 split, prints a side-by-side comparison of Precision, Recall, F1-Score, and AUC-ROC, and saves three plots to `model/`.

```bash
python model/evaluation.py
```

Output plots saved to `model/`:

| File | Contents |
|---|---|
| `eval_roc_curves.png` | ROC curves for all 5 models overlaid |
| `eval_confusion_matrices.png` | Confusion matrix for each model |
| `eval_metric_comparison.png` | Bar chart comparing all metrics |

"""
Comprehensive Model Evaluation — Bank Telemarketing Subscription Prediction
Models: Logistic Regression, Decision Tree, Random Forest, XGBoost, Neural Network
Metrics: Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split , StratifiedKFold
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, roc_curve, ConfusionMatrixDisplay
)
import tensorflow as tf


from models.decision_tree import dt  # Reuse the GridSearchCV setup from decision_tree.py
from models.logistic import lr  # Reuse the Logistic Regression setup from logistic.py
from models.random_forest import rf  # Reuse the Random Forest setup from random_forest.py
from models.XGBoost import xgb  # Reuse the XGBoost setup from XGBoost.py
from models.neural_network import nn_build_and_compile, nn_fit, encode_months  # Reuse the Neural Network setup from neural_network.py

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "master_features.csv")
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))

# ── Load & clean data ─────────────────────────────────────────────────────────
data = pd.read_csv(DATA_PATH)
# XGBoost requires clean column names (no special chars)
data.columns = [
    re.sub(r'[\[\]<>()]', '', c).strip().replace('  ', ' ').replace(' ', '_')
    for c in data.columns
]

TARGET = "target_y"
TOP_FEATURES = [
    'macro_economic_pca_1', 'month_may', 'month_apr',
    'month_oct', 'contact_cellular', 'prior_engagement',
    'job_grouped_not_working'
]

X_all = data.drop(columns=[TARGET])
y     = data[TARGET]

# ── Shared train / test split (stratified to preserve 88/12 class ratio) ─────

random_state = 1244
X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=random_state, stratify=y
)
X_train = X_train_all[TOP_FEATURES]
X_test  = X_test_all[TOP_FEATURES]

print(f"Dataset: {len(data):,} rows  |  Train: {len(y_train):,}  |  Test: {len(y_test):,}")
print(f"Positive rate — Train: {y_train.mean():.3f}  |  Test: {y_test.mean():.3f}\n")

# Global Variables for model training

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

neg_prop, pos_prop = y_train.value_counts(normalize=True).sort_index()
class_weights = {0: 1 / neg_prop, 1: 1 / pos_prop}

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos = neg_count / pos_count

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1/5  Logistic Regression")
print("=" * 60)

lr = lr(class_weights, random_state)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
lr_prob = lr.predict_proba(X_test)[:, 1]

# ══════════════════════════════════════════════════════════════════════════════
# 2. DECISION TREE  (GridSearchCV, class_weight='balanced')
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2/5  Decision Tree  (GridSearchCV)")
print("=" * 60)

dt_grid = dt(class_weight= class_weights, random_state=random_state, cv=cv)  # Reuse the GridSearchCV setup from decision_tree.py

dt_grid.fit(X_train, y_train)

best_dt = dt_grid.best_estimator_
dt_pred = best_dt.predict(X_test)
dt_prob = best_dt.predict_proba(X_test)[:, 1]

print(f"Best params : {dt_grid.best_params_}")
print(f"Best CV AUC : {dt_grid.best_score_:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. RANDOM FOREST  (RandomizedSearchCV, class_weight='balanced')
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3/5  Random Forest  (RandomizedSearchCV)")
print("=" * 60)


rf_search = rf(class_weights, random_state, cv)  # Reuse the RandomizedSearchCV setup from random_forest.py

rf_search.fit(X_train, y_train)

best_rf = rf_search.best_estimator_
rf_pred = best_rf.predict(X_test)
rf_prob = best_rf.predict_proba(X_test)[:, 1]

print(f"Best params : {rf_search.best_params_}")
print(f"Best CV AUC : {rf_search.best_score_:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. XGBOOST  (all features, scale_pos_weight for imbalance)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4/5  XGBoost  (all features)")
print("=" * 60)


print(f"scale_pos_weight (train only): {scale_pos:.4f}")

xgb_clf = xgb(scale_pos_weight=scale_pos, random_state=random_state)

xgb_clf.fit(X_train_all, y_train)

xgb_pred = xgb_clf.predict(X_test_all)
xgb_prob = xgb_clf.predict_proba(X_test_all)[:, 1]

# ══════════════════════════════════════════════════════════════════════════════
# 5. NEURAL NETWORK  (cyclical month encoding, class_weight dict)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5/5  Neural Network  (TensorFlow/Keras)")
print("=" * 60)

X_train_nn, X_test_nn = encode_months(TOP_FEATURES, X_train, X_test)  # Reuse the month encoding function from neural_network.py

nn_model = nn_build_and_compile(optimizer='adam', loss='binary_crossentropy')  # Reuse the model architecture function from neural_network.py
# the nn_build_and_compile function uses metrics=[keras.metrics.AUC(name='auc') internally, so we don't need to specify it again here

nn_fit(nn_model, X_train_nn, y_train, class_weights, epochs=100, batch_size=32)  # Reuse the fitting function from neural_network.py

nn_prob = nn_model.predict(X_test_nn, verbose=0).flatten()
nn_pred = (nn_prob > 0.5).astype(int)

print("Neural Network trained successfully.")

# ══════════════════════════════════════════════════════════════════════════════
# COLLECT ALL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def metrics(y_true, y_pred, y_prob):
    return {
        'Precision': round(float(precision_score(y_true, y_pred)), 4),
        'Recall':    round(float(recall_score(y_true, y_pred)),    4),
        'F1-Score':  round(float(f1_score(y_true, y_pred)),        4),
        'AUC-ROC':   round(float(roc_auc_score(y_true, y_prob)),   4),
        'y_pred':    y_pred,
        'y_prob':    y_prob,
    }

results = {
    'Logistic Regression': metrics(y_test, lr_pred,  lr_prob),
    'Decision Tree':       metrics(y_test, dt_pred,  dt_prob),
    'Random Forest':       metrics(y_test, rf_pred,  rf_prob),
    'XGBoost':             metrics(y_test, xgb_pred, xgb_prob),
    'Neural Network':      metrics(y_test, nn_pred,  nn_prob),
}

# ══════════════════════════════════════════════════════════════════════════════
# DETAILED CLASSIFICATION REPORTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 60)
print("DETAILED CLASSIFICATION REPORTS")
print("=" * 60)

for name, r in results.items():
    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")
    print(classification_report(
        y_test, r['y_pred'],
        target_names=['Unsuccessful (0)', 'Successful (1)']
    ))

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
METRIC_COLS = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC']
summary = pd.DataFrame(
    {name: {m: r[m] for m in METRIC_COLS} for name, r in results.items()}
).T

print("\n\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
print(summary.to_string())
print()
for col in METRIC_COLS:
    best_name = summary[col].idxmax()
    print(f"  Best {col:<12}: {best_name}  ({summary.loc[best_name, col]:.4f})")

# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION 1 — ROC Curves (all models on one plot)
# ══════════════════════════════════════════════════════════════════════════════
COLORS = ['steelblue', 'darkorange', 'forestgreen', 'crimson', 'mediumpurple']

fig, ax = plt.subplots(figsize=(8, 6))
for (name, r), color in zip(results.items(), COLORS):
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    ax.plot(fpr, tpr, label=f"{name}  (AUC = {r['AUC-ROC']:.4f})", color=color, lw=2)
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — All Models')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eval_roc_curves.png"), dpi=150)
plt.show()
print("Saved: eval_roc_curves.png")

# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION 2 — Confusion Matrices
# ══════════════════════════════════════════════════════════════════════════════
n_models = len(results)
fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))

for ax, (name, r), color in zip(axes, results.items(), COLORS):
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, r['y_pred']),
        display_labels=['Unsuccessful', 'Successful']
    ).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('Actual', fontsize=9)
    ax.tick_params(labelsize=8)

fig.suptitle('Confusion Matrices — All Models', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "eval_confusion_matrices.png"),
    dpi=150, bbox_inches='tight'
)
plt.show()
print("Saved: eval_confusion_matrices.png")

# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION 3 — Metric Comparison Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
model_names = list(results.keys())
x = np.arange(len(model_names))

for ax, metric in zip(axes.flat, METRIC_COLS):
    vals = [results[name][metric] for name in model_names]
    bars = ax.bar(x, vals, color=COLORS, edgecolor='white', linewidth=0.8)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha='right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f'{val:.3f}',
            ha='center', va='bottom', fontsize=8.5, fontweight='bold'
        )
    best_idx = int(np.argmax(vals))
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

fig.suptitle('Model Performance Comparison\n(gold border = best per metric)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eval_metric_comparison.png"), dpi=150)
plt.show()
print("Saved: eval_metric_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# BEST MODEL RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
best_auc = summary['AUC-ROC'].idxmax()
best_f1  = summary['F1-Score'].idxmax()
best_rec = summary['Recall'].idxmax()
best_pre = summary['Precision'].idxmax()

print("\n\n" + "=" * 60)
print("BEST MODEL RECOMMENDATION")
print("=" * 60)
print(f"\n  Best AUC-ROC  : {best_auc:<22} ({summary.loc[best_auc,  'AUC-ROC']:.4f})")
print(f"  Best F1-Score : {best_f1:<22} ({summary.loc[best_f1,  'F1-Score']:.4f})")
print(f"  Best Recall   : {best_rec:<22} ({summary.loc[best_rec, 'Recall']:.4f})")
print(f"  Best Precision: {best_pre:<22} ({summary.loc[best_pre, 'Precision']:.4f})")

print("""
  Context (imbalanced telemarketing data, ~12% positive class):

  • AUC-ROC  is the primary metric — it measures overall discriminative
    ability regardless of decision threshold, ideal for imbalanced classes.

  • Recall   matters most when missing a potential subscriber is costly
    (false negatives = lost revenue).

  • Precision matters when call-centre resources are limited and wasted
    calls are costly (false positives = wasted effort).

  • F1-Score balances both for a single-number comparison.
""")
print(f"  ★  Recommended model: {best_auc}  (highest AUC-ROC)")
print("=" * 60)

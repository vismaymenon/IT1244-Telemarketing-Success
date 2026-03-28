import re
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay

# ── 0. Clean column names (do this ONCE on X and df) ───────────────────────
clean = lambda cols: [
    re.sub(r'[\[\]<>()]', '', c).strip().replace('  ', ' ').replace(' ', '_')
    for c in cols
]
X.columns = clean(X.columns)
df.columns = clean(df.columns)


# ── 1. Stratified train/test split ─────────────────────────────────────────
#    stratify=y ensures both splits mirror the 88/12 class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1244, stratify=y   # ← stratify keeps ratio in both halves
)


# ── 2. Compute scale_pos_weight from TRAINING labels only ──────────────────
#    Never use global counts — that would leak test-set label info
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale = neg_count / pos_count
print(f"scale_pos_weight (train only): {scale:.4f}")


# ── 3. Define the model with imbalance correction ──────────────────────────
xgb_clf = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    objective='binary:logistic',    # binary → your target is binary (0 = Unsuccessful, 1 = Successful) 
                                    # logistic → use logistic regression as the underlying prediction model
                                    # The model outputs a probability between 0 and 1 (e.g., 0.73 = 73% chance of success)
                                    # Internally it minimises cross-entropy loss
    eval_metric='logloss',          # logloss (log loss) penalises confident wrong predictions more heavily than uncertain ones
                                    # A lower logloss = better model
    n_jobs=-1,
    random_state=42
)


# ── 4. Cross-validate BEFORE final fit (model validation step) ─────────────
#    This tells you how well your design generalises — use X_train/y_train only
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

cv_scores = cross_val_score(
    xgb_clf, X_train, y_train,      # ← CV runs on training data only, test set stays hidden
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1
)
print(f"\nCV Mean AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")


# ── 5. Refit final model on full training set ──────────────────────────────
#    After CV confirms your config is sound, train once on all of X_train
xgb_clf.fit(X_train, y_train)


# ── 6. Evaluate on held-out test set ──────────────────────────────────────
y_pred  = xgb_clf.predict(X_test)
y_proba = xgb_clf.predict_proba(X_test)[:, 1]

print("\nClassification Report (XGBoost + scale_pos_weight):")
print(classification_report(y_test, y_pred, target_names=["Unsuccessful", "Successful"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("XGBoost ROC Curve – Telemarketing Success")
plt.tight_layout()
plt.show()

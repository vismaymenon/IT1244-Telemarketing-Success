'''
Random Forest Classifier for Bank Telemarketing Success Prediction

Random Forest builds on Decision Trees by:
- Training multiple trees on bootstrap samples (bagging)
- Each split considers a random subset of features → reduces correlation between trees
- Aggregating predictions reduces variance and overfitting vs a single Decision Tree
- Expected to match or exceed Decision Tree AUC (baseline ~0.8022)
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data and prepare training and testing sets
data = pd.read_csv("data/master_features.csv")
X = data.drop(columns=["target_y"])
y = data["target_y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1244)

# Select top features based on feature importance analysis in preprocessing.ipynb
top_features = [
    'macro_economic_pca_1', 'month_may', 'month_apr',
    'month_oct', 'contact_cellular', 'prior_engagement',
    'job_grouped_not_working'
]

# Prepare final training and testing sets with selected features
X_train_final = X_train[top_features]
X_test_final = X_test[top_features]

# Tune via randomised search (faster than full grid search for RF)
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 5, 6, 7, None],
    'min_samples_leaf': [1, 5, 10, 20],
    'max_features': ['sqrt', 'log2']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1244)

random_search = RandomizedSearchCV(
    RandomForestClassifier(class_weight= 'balanced', random_state=1244, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=30,
    scoring='roc_auc',
    cv=cv,
    random_state=1244,
    n_jobs=-1
)
random_search.fit(X_train_final, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV AUC: {random_search.best_score_:.4f}")

# Refit best model on full training set
best_rf = random_search.best_estimator_

# Evaluate on test set
y_prob = best_rf.predict_proba(X_test_final)[:, 1]
y_pred = best_rf.predict(X_test_final)

print(classification_report(y_test, y_pred, target_names=['Unsuccessful', 'Successful']))
print(f"Test AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Unsuccessful', 'Successful'])
plt.title("Random Forest — Confusion Matrix")
plt.tight_layout()
plt.savefig("modules/models/images/rf/rf_cm.png")
plt.show()

# Feature importances (mean decrease in impurity across all trees)
importances = pd.Series(best_rf.feature_importances_, index=top_features).sort_values(ascending=True)
std = np.std([tree.feature_importances_ for tree in best_rf.estimators_], axis=0)
std_sorted = std[importances.index.map(lambda x: top_features.index(x))]

fig, ax = plt.subplots()
importances.plot(kind='barh', xerr=std_sorted, ax=ax, title='Random Forest — Feature Importances')
ax.set_xlabel('Mean Decrease in Impurity')
plt.tight_layout()
plt.savefig("modules/models/images/rf/rf_feature_importance.png")
plt.show()

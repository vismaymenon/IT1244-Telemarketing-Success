'''
Decision Tree Classifier for Bank Telemarketing Success Prediction

Key observations from EDA/feature selection:
- macro_economic_pca_1 alone explains 73.5% of feature importance
- Relationships are threshold-based (e.g. "if macro_economic_pca_1 > X, predict success")
- Decision Trees naturally excel at this structure → expected to outperform neural network
- CV AUC with 7 selected features: 0.8022 ± 0.0038 (vs NN AUC of 0.7908)
'''

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

# Read data and prepare training and testing sets
data = pd.read_csv("./data/master_features.csv")
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

# Tune max_depth and min_samples_leaf via cross-validated grid search
param_grid = {
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_leaf': [1, 5, 10, 20],
    'criterion': ['gini', 'entropy']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1244)
grid_search = GridSearchCV(
    DecisionTreeClassifier(class_weight='balanced', random_state=1244),
    param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1
)
grid_search.fit(X_train_final, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV AUC: {grid_search.best_score_:.4f}")

# Refit best model on full training set
best_dt = grid_search.best_estimator_

# Evaluate on test set
y_prob = best_dt.predict_proba(X_test_final)[:, 1]
y_pred = best_dt.predict(X_test_final)

print(classification_report(y_test, y_pred, target_names=['Unsuccessful', 'Successful']))
print(f"Test AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Unsuccessful', 'Successful'])
plt.title("Decision Tree — Confusion Matrix")
plt.tight_layout()
plt.savefig("./models/dt_cm.png")
plt.show()

# Visualise the tree (limited depth for readability)
plt.figure(figsize=(20, 8))
plot_tree(
    best_dt,
    feature_names=top_features,
    class_names=['Unsuccessful', 'Successful'],
    filled=True,
    max_depth=3,         # Show top 3 levels only for clarity
    impurity=False,
    proportion=True
)
plt.title("Decision Tree Structure (top 3 levels)")
plt.tight_layout()
plt.savefig("./models/dt_tree.png")
plt.show()

# Feature importances
importances = pd.Series(best_dt.feature_importances_, index=top_features).sort_values(ascending=True)
importances.plot(kind='barh', title='Decision Tree — Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig("./models/dt_feature_importance.png")
plt.show()

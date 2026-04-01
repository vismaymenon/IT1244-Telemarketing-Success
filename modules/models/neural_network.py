'''
- The dominant feature (macro_economic_pca_1) explains 73.5% of importance alone
- Relationships are likely more threshold-based than continuous

Decision Trees are naturally good at threshold-based splits — 
"if macro_economic_pca_1 > X, predict positive." 
Neural networks are designed to learn complex non-linear relationships, 
which may be overkill for this problem. Sometimes simpler models win on simpler problems.
~ Claude


This is becaue the AUC for for the nn is 0.7908, which is lower than the AUC of 0.8022 for the DT
'''

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

month_number_mapping = {
    'month_jan': 1,
    'month_feb': 2,
    'month_mar': 3,
    'month_apr': 4,
    'month_may': 5,
    'month_jun': 6,
    'month_jul': 7,
    'month_aug': 8,
    'month_sep': 9,
    'month_oct': 10,
    'month_nov': 11,
    'month_dec': 12
}
# Prepare final training and testing sets with selected features
X_train_final = X_train[top_features]
X_test_final = X_test[top_features]

negative_prop, positive_prop = y_train.value_counts(normalize=True)
class_weights = {0: 1 /negative_prop, 1: 1 /positive_prop}  # Adjust class weights to handle imbalance


#Handle cyclical month encoding
def add_cyclical_month_encoding(df, month_col):
    month_num = month_number_mapping[month_col]
    df[f'{month_col}_sin'] = np.sin(2 * np.pi * month_num / 12)* df[month_col]
    df[f'{month_col}_cos'] = np.cos(2 * np.pi * month_num / 12)* df[month_col]
    df.drop(columns=[month_col], inplace=True)
    return df

def encode_months(top_features, X_train, X_test):
    X_train_final = X_train.copy()
    X_test_final = X_test.copy()
    months_to_encode = [col for col in top_features if col.startswith('month_')]
    for month_col in months_to_encode:
        X_train_final = add_cyclical_month_encoding(X_train_final, month_col)
        X_test_final = add_cyclical_month_encoding(X_test_final, month_col)
    return X_train_final, X_test_final

X_train_final, X_test_final = encode_months(top_features, X_train, X_test)

print(X_train_final.shape)
print(X_train_final.columns.tolist())

# Define the model architecture
def nn_build_and_compile(optimizer, loss):
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train_final.shape[1],)),
        keras.layers.Dense(7, activation='relu'), # 2/3 of the number of features
        keras.layers.Dropout(0.2),
        keras.layers.Dense(7, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid') # Binary classification output
    ])
    model.compile(optimizer=optimizer, loss=loss, metrics=[keras.metrics.AUC(name='auc')])
    return model

#Compile with adam + binary_crossentropy + AUC metric
model = nn_build_and_compile(optimizer='adam', loss='binary_crossentropy')

# Fit with early stopping + validation split + class weights
def nn_fit(model, X_train, y_train, class_weights, epochs, batch_size):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, 
                        validation_split=0.2, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        class_weight=class_weights,
                        callbacks=[early_stopping])
    return history

history = nn_fit(model, X_train_final, y_train, class_weights, epochs=100, batch_size=32)
# Evaluate on test set
y_prob = model.predict(X_test_final)
y_pred = (y_prob > 0.5).astype(int)

# Print classification report and AUC-ROC
print(classification_report(y_test, y_pred, target_names=['Unsuccessful', 'Successful']))
print(f"Test AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Unsuccessful', 'Successful'])
plt.show()

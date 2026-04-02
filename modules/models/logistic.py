import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, classification_report, roc_auc_score

#data prep
cwd = Path.cwd()
data_path = cwd / 'data' / 'master_features.csv'
cleandf = pd.read_csv(data_path)
xvals = cleandf[['macro_economic_pca_1', 'month_may', 'month_apr', 'month_oct', 'contact_cellular', 'prior_engagement', 'job_grouped_not_working']]
yvals = cleandf["target_y"]

#train test split
xtrain, xtest, ytrain, ytest = train_test_split(xvals, yvals, test_size=0.2, random_state=1244, stratify=yvals)
negative_prop, positive_prop = yvals.value_counts(normalize=True)
class_weights = {0: 1 /negative_prop, 1: 1 /positive_prop}  # Adjust class weights to handle imbalance

#regression
model = LogisticRegression(class_weight=class_weights, random_state=1244)
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

#evaluation
cm = confusion_matrix(ytest, ypred)
print(cm)
acc = accuracy_score(ytest, ypred)
print(acc)
f1 = f1_score(ytest, ypred)
print(f1)
recall = recall_score(ytest, ypred)
print(recall)
print(classification_report(ytest, ypred, target_names=['Unsuccessful', 'Successful']))
roc_auc = roc_auc_score(ytest, ypred)
print(roc_auc)

import pandas as pd

#data prep
cleandf = pd.read_csv("master_features.csv")
xvals = cleandf[['macro_economic_pca_1', 'month_may', 'month_apr', 'month_oct', 'contact_cellular', 'prior_engagement', 'job_grouped_not_working']]
yvals = cleandf["target_y"]

#train test split
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(xvals, yvals, test_size=0.2, random_state=1244, stratify=yvals)

#regression
from sklearn.linear_model import LogisticRegression
classifer = LogisticRegression()
classifer.fit(xtrain, ytrain)
ypred = classifer.predict(xtest)

#evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, classification_report, roc_auc_score
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

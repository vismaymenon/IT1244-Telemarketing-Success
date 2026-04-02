# How to Run the Models

## Prerequisites
All commands should be run from the **project root directory**. To check run

``` bash
pwd
```
Ensure the present working directory is the **root folder**. If it is not change the working directory to be in the root folder.

**Step 1 — Install dependencies** (from the project root):

```bash
pip install -r requirements.txt
```

**Step 2 — Generate the preprocessed feature matrix:**

Open `preprocessing.ipynb` in Jupyter and run all cells from top to bottom.
This produces `data/master_features.csv`, which all model scripts depend on.

```bash
jupyter nbconvert --to notebook --execute modules/preprocessing.ipynb
```

This line will execute the preprocessing file and produce a new notebook named `preprocessing.nbconvert.ipynb` which will containt he results

**However**, if you wish to save the outputs in the original file rather than create a new one, run the following command instead.

``` bash
jupyter nbconvert --to notebook --execute --inplace modules/preprocessing.ipynb
```

> If `data/master_features.csv` does not exist, the scripts below will fail.

---

## Running the Models

All commands should be run from the **project root directory** (not inside `model/`).

### Individual Model Scripts
**There is no need to run these.** These scripts are not needed to run `master.ipynb`. Skip to the next section to run the `master.ipynb` that will run all 5 models and provide the relevant perfomance metrics for each model. `master.ipynb` will evaluate the models as well.

```bash
python modules/models/logistic.py        # Logistic Regression (baseline)
python modules/models/decision_tree.py   # Decision Tree with GridSearchCV
python modules/models/random_forest.py   # Random Forest with RandomizedSearchCV
python modules/models/XGBoost.py         # XGBoost with scale_pos_weight
python modules/models/neural_network.py  # Neural Network (TensorFlow/Keras)
```
Decision Tree, Random Forest, XGBoost and Neural Networks produce images as part of their evaluation. The images will be stored in `modules/models/images`.
### Full 5-Model Comparison

Trains all five models on the same stratified 80/20 split, prints a side-by-side comparison of Precision, Recall, F1-Score, and AUC-ROC. This file will provide the final run 

```bash
jupyter nbconvert --to notebook --execute modules/master.ipynb
```
This line will execute the master file and produce a new notebook named `master.nbconvert.ipynb` which will containt he results, metrics and evalutaions.

**However**, if you wish to save the outputs in the original file rather than create a new one, run the following command instead.

``` bash
jupyter nbconvert --to notebook --execute --inplace modules/master.ipynb
```
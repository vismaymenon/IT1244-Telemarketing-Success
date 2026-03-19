# Bank Telemarketing Campaign Dataset Documentation

## Overview and Problem Statement
Direct marketing campaigns are a critical tool for customer acquisition in the banking sector. Predicting customer response to marketing offers allows banks to optimize resource allocation and improve campaign effectiveness. In this project you are tasked with predicting whether a client will subscribe to a term deposit product based on demographic information, previous campaign history, and socioeconomic context.


To achieve this, you are provided with approximately 40,000 customer records from a Portuguese banking institution's telemarketing campaigns.


## Dataset File Naming Convention

The dataset is provided as a single CSV file: `dataset.csv`

This file contains customer information and campaign outcome labels.

## Data Structure

### CSV File Format

The dataset contains approximately 40,000 rows with 21 columns (20 features + 1 target).

### Feature Categories

**1. Customer Demographics (7 features):**

- `age` : Age in years (numeric)
- `job` : Type of job (categorical)
  - Values: "admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"
- `marital` : Marital status (categorical)
  - Values: "divorced", "married", "single", "unknown"
  - Note: "divorced" includes widowed
- `education` : Education level (categorical)
  - Values: "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree", "unknown"
- `default` : Has credit in default? (categorical, encoded)
  - 0: No, 1: Yes, -1: Unknown
- `housing` : Has housing loan? (categorical, encoded)
  - 0: No, 1: Yes, -1: Unknown
- `loan` : Has personal loan? (categorical, encoded)
  - 0: No, 1: Yes, -1: Unknown

**2. Current Campaign Contact Information (3 features):**

- `contact` : Contact communication type (categorical)
  - Values: "cellular", "telephone"
- `month` : Last contact month of year (categorical)
  - Values: "jan", "feb", "mar", ..., "nov", "dec"
- `day_of_week` : Last contact day of the week (categorical)
  - Values: "mon", "tue", "wed", "thu", "fri"

**3. Campaign-Specific Features (4 features):**

- `duration` : Last contact duration in seconds (numeric)
  - **Important Note**: This feature is highly predictive but not available before a call is made. For realistic predictive modeling, this feature should be excluded. Include it only for benchmark purposes.
- `campaign` : Number of contacts performed during this campaign for this client (numeric)
  - Includes the last contact
- `pdays` : Days since client was last contacted from a previous campaign (numeric)
  - 999 means client was not previously contacted
- `previous` : Number of contacts performed before this campaign for this client (numeric)

**4. Socioeconomic Context Attributes (5 features):**

These are external indicators that reflect economic conditions:

- `emp.var.rate` : Employment variation rate - quarterly indicator (numeric)
- `cons.price.idx` : Consumer price index - monthly indicator (numeric)
- `cons.conf.idx` : Consumer confidence index - monthly indicator (numeric)
- `euribor3m` : Euribor 3-month rate - daily indicator (numeric)
- `nr.employed` : Number of employees - quarterly indicator (numeric)

**5. Target Variable:**

- `y` : Has the client subscribed to a term deposit? (categorical, encoded)
  - 0: No
  - 1: Yes
  - -1: Unknown (if present)

## Data Quality Issues

### Missing Values
Missing values are encoded as "unknown" in categorical variables rather than NULL/NaN. You'll need to handle these explicitly during preprocessing.

### Class Imbalance
The dataset is imbalanced with fewer positive outcomes (subscriptions) than negative outcomes (rejections), reflecting real-world campaign response rates.

### Duration Feature Caveat
The `duration` feature is highly correlated with the target because:
- If duration = 0, the outcome is definitely "no" (call not completed)
- Longer calls often indicate customer interest

However, duration is unknown before making the call, so including it creates data leakage. For realistic modeling, exclude this feature.

## Suggested Train-Test Split

Perform appropriate train-test splitting for evaluation:
- Use 5-fold cross-validation for model development and hyperparameter tuning
- Reserve a held-out test set for final performance assessment
- Consider stratified sampling to maintain class distribution

## Suggestions & Comments

Here we shall provide some directions to help kickstart your project. Nevertheless, you are not required to answer these questions and encouraged to explore other questions that might be interesting to you.

### Feature Preprocessing
Different feature types require different preprocessing:
- **Categorical variables**: Should you use one-hot encoding, label encoding, or target encoding?
- **Numerical variables**: Do you need scaling or normalization?
- **Unknown values**: Treat as a separate category, impute, or drop?
- **Temporal features**: Can you extract additional information from month/day combinations?

### Feature Selection and Relevance
Not all features may be equally informative:
- Which features are most predictive of subscription?
- Do socioeconomic indicators add value beyond customer demographics?
- Are there redundant or highly correlated features?

### Handling Class Imbalance
The imbalanced nature of the data affects modeling:
- Should you use class weighting in your model?
- Do resampling techniques (SMOTE, undersampling) help?
- Which evaluation metrics are appropriate? (Precision, Recall, F1, AUC-ROC)
- Is accuracy misleading for this problem?

### Campaign Strategy Insights
Beyond prediction, can you extract actionable insights?
- Which customer segments are most likely to subscribe?
- What is the optimal contact frequency (campaign count)?
- Do previous campaign contacts help or hurt conversion?
- Is there an optimal time (month, day of week) to contact customers?

### Temporal and Economic Effects
The socioeconomic indicators provide context:
- How do economic conditions affect subscription rates?
- Are customers more receptive during certain economic periods?
- Can you identify whether campaigns performed during different months have different outcomes?

### Evaluation Strategy
For imbalanced classification, consider multiple metrics:
- **Precision**: Of predicted subscriptions, how many are correct?
- **Recall (Sensitivity)**: Of actual subscriptions, how many are caught?
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Threshold-independent performance measure
- **Confusion Matrix**: Understand specific error types

### Business Impact
Frame model performance in business terms:
- What is the cost of a false positive (wasted call)?
- What is the cost of a false negative (missed customer)?
- Can you optimize for business value rather than just accuracy?
- How many calls can you save by targeting high-probability customers?


# Predictive Maintenance AI

A Streamlit machine-learning application that estimates whether a machine is
likely to be in the **machine-failure class** from its operating conditions and
sensor readings. The project includes data download, preprocessing, model
training and comparison, evaluation, and an interactive prediction dashboard.

> **Educational prototype:** This project uses the AI4I 2020 synthetic
> predictive-maintenance dataset. It is not connected to real equipment or live
> sensors, and its predictions and risk-score bands are not validated for
> operational maintenance decisions.

## Contents

- [What the project does](#what-the-project-does)
- [Dataset](#dataset)
- [Training and model selection](#training-and-model-selection)
- [Saved test-set results](#saved-test-set-results)
- [Run the project](#run-the-project)
- [Using the application](#using-the-application)
- [Project structure](#project-structure)
- [Limitations and next steps](#limitations-and-next-steps)

## What the project does

The project learns a binary classification task:

- **0 — No machine failure**
- **1 — Machine failure**

The training script compares two classifiers and saves the one with the higher
test-set F1-score. The Streamlit app loads that saved pipeline to classify
sensor readings entered by a user, show the model's estimated failure score,
and display model and dataset information.

The model estimates the class for the supplied readings. It does **not** predict
when a failure will happen or estimate remaining useful life.

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset** from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).
The downloader fetches UCI dataset ID `601` and writes it to
`data/ai4i2020.csv`.

The checked-in dataset contains **10,000 rows** and **12 columns**. The UCI
source may include identifier columns in some versions; the training script
excludes `UDI` and `Product ID` if they are present. The checked-in CSV does not
contain those columns.

| Column | Meaning | Used as a model input? |
|---|---|---|
| `Type` | Product quality/type category (`L`, `M`, or `H`) | Yes |
| `Air temperature` | Air temperature in kelvin | Yes |
| `Process temperature` | Process temperature in kelvin | Yes |
| `Rotational speed` | Rotational speed in rpm | Yes |
| `Torque` | Torque in Nm | Yes |
| `Tool wear` | Tool wear in minutes | Yes |
| `Machine failure` | Overall binary target label | Target |
| `TWF` | Tool wear failure mode | No; failure-mode/target-related |
| `HDF` | Heat dissipation failure mode | No; failure-mode/target-related |
| `PWF` | Power failure mode | No; failure-mode/target-related |
| `OSF` | Overstrain failure mode | No; failure-mode/target-related |
| `RNF` | Random failure mode | No; failure-mode/target-related |

The dataset has **9,661 non-failure rows (96.61%)** and **339 failure rows
(3.39%)**. This class imbalance is why accuracy alone is not enough to evaluate
the classifiers.

## Training and model selection

Run `train_model.py` to perform the following steps:

1. Read `data/ai4i2020.csv` and use `Machine failure` as the target.
2. Exclude identifiers (`UDI`, `Product ID`) if present, and exclude
   failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) from the input
   features. This avoids giving the model identifiers or closely target-related
   failure-mode labels.
3. Split the data into **80% training and 20% test**, using
   `train_test_split(test_size=0.20, random_state=42, stratify=y)`.
   - Training: **8,000 rows** (7,729 non-failures and 271 failures).
   - Test: **2,000 rows** (1,932 non-failures and 68 failures).
   - Stratification preserves approximately the same failure proportion in
     both sets; seed `42` makes the split reproducible.
4. Fit preprocessing as part of each model's scikit-learn pipeline:
   - Numeric features: fill missing values with the median, then standardize
     using `StandardScaler`.
   - Categorical features: fill missing values with the most frequent value,
     then one-hot encode with unknown categories ignored.
5. Train and evaluate the following classifiers:

   | Model | Configuration | Why it is included |
   |---|---|---|
   | Logistic Regression | `max_iter=2000`, `class_weight="balanced"` | A comparatively simple, interpretable linear baseline. Balanced class weights account for the minority failure class. |
   | Random Forest | 250 trees, `min_samples_leaf=2`, `class_weight="balanced"` | A tree ensemble can learn nonlinear relationships and feature interactions. Balanced class weights help address the rare failure class. |

6. Calculate accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix,
   and classification report for each model on the test set.
7. Select the model with the highest **F1-score**, then save the complete
   preprocessing-and-model pipeline to `models/model.joblib`. Save the selected
   model name, feature list, split configuration, and comparison metrics to
   `models/metrics.json`.

### Why use F1-score?

With only 3.39% failure examples, a classifier could achieve high accuracy by
mostly predicting “no failure.” F1-score combines precision (how many predicted
failures were actual failures) and recall (how many actual failures were found),
making it more informative for comparing these imbalanced classifiers. The
application also reports precision, recall, accuracy, and ROC-AUC so that
trade-offs remain visible.

## Saved test-set results

These are the results currently recorded in `models/metrics.json`. Values are
from the held-out 20% split; the app displays this comparison on its **Model
Performance** page.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8245 | 0.1418 | 0.8235 | 0.2419 | 0.9070 |
| Random Forest *(selected)* | 0.9745 | 0.5955 | 0.7794 | 0.6752 | 0.9682 |

The Random Forest is selected because its F1-score is higher in this recorded
run. Its precision/recall trade-off is important: some predicted failures are
false alarms, and some actual failures are missed.

**Evaluation caveat:** The current script uses the same test split both to
compare/select the models and to report their metrics. That can make the
reported results optimistic; the test set is no longer fully untouched after
model selection. A stronger evaluation would use a separate validation set or
cross-validation to select the model, then evaluate the final choice once on a
reserved test set.

## Run the project

Commands below are shown for Windows PowerShell. Run them from the repository
root.

### 1. Create and activate a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the environment's Python directly as shown
below, or follow your local PowerShell execution-policy guidance.

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Get the dataset

The CSV is included in this repository. If it is missing, download it with:

```powershell
python download_data.py
```

This script uses `ucimlrepo` to fetch dataset ID `601` and saves the resulting
CSV to `data/ai4i2020.csv`; an internet connection is required for this step.

### 4. Train and save the model

```powershell
python train_model.py
```

Training prints the model metrics, confusion matrix, and classification report,
then creates or replaces `models/model.joblib` and `models/metrics.json`.

### 5. Start the dashboard

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit in your browser. The app requires the
saved model and metrics files; train the model first if either is missing.

## Using the application

The sidebar provides four sections:

- **Dashboard:** Dataset record counts, observed failure rate, failure
  distribution, and a sensor chart.
- **Predict Failure:** Enter machine type and sensor readings to see the
  predicted class and model score. The submitted row can be downloaded as a
  CSV.
- **Model Performance:** Compare the saved test metrics and see which model was
  selected by F1-score.
- **Dataset Explorer:** Preview records, view descriptive statistics and
  missing-value counts, and download the dataset.

The prediction score is the model's estimated probability for the failure
class. The UI divides scores below `0.30`, from `0.30` to below `0.60`, and
`0.60` or higher into illustrative LOW, MEDIUM, and HIGH score bands. These
bands are not calibrated or validated maintenance thresholds, and should not
be treated as real-world risk levels.

## Project structure

```text
.
├── app.py                 # Streamlit dashboard and interactive prediction
├── download_data.py       # Downloads the AI4I 2020 dataset from UCI
├── train_model.py         # Preprocessing, training, comparison, evaluation
├── requirements.txt       # Python package dependencies
├── data/
│   └── ai4i2020.csv       # Dataset used for training and exploration
└── models/
    ├── model.joblib       # Saved selected scikit-learn pipeline
    └── metrics.json       # Saved model comparison and split metadata
```

## Limitations and next steps

- The source dataset is synthetic and does not establish performance on a
  particular factory, machine, sensor setup, or real operating conditions.
- Failure is rare in the data, so false alarms and missed failures both matter;
  review precision and recall in addition to accuracy.
- Model selection currently reuses the test set. Add a validation strategy and
  keep an untouched final test set for less biased evaluation.
- The LOW/MEDIUM/HIGH bands are illustrative score ranges, not validated
  decision thresholds. Probability calibration and domain-specific threshold
  selection would be needed before making risk claims.
- The app accepts one set of sensor values at a time; it does not consume live
  telemetry, model temporal trends, or predict time-to-failure.

## Dependencies

See [`requirements.txt`](./requirements.txt) for the package list. The project
uses pandas and NumPy for data handling, scikit-learn for preprocessing and
classification, joblib for model persistence, and Streamlit and Matplotlib for
the dashboard. `ucimlrepo` is used by the dataset downloader.
# Machine Failure Prediction Using Machine Learning

A Machine Learning-based Predictive Maintenance System that predicts whether a machine is likely to fail based on its sensor readings. The project uses historical machine data, compares two machine learning models, and provides predictions through an interactive Streamlit web application.

## 1. Project Overview

Machine failures can cause unexpected downtime, production delays, and increased maintenance costs. Traditional maintenance often depends on fixed schedules or repairs after a machine breaks down.

This project demonstrates how Machine Learning can help identify potential machine failures by analyzing sensor readings such as temperature, rotational speed, torque, and tool wear.

The application allows users to enter machine sensor values and receive a prediction:

- **No Failure:** The model predicts that the machine will not fail.
- **Failure:** The model predicts that the machine may experience a failure.

This is an educational prototype based on a historical dataset. It does not currently connect to physical machines or live sensors.

## 2. Project Objectives

- Build a Machine Learning model for machine failure classification.
- Analyze machine sensor data using Python.
- Train and compare Logistic Regression and Random Forest.
- Automatically select the better-performing model based on F1-score.
- Create an interactive web application using Streamlit.
- Display model performance metrics and dataset information.
- Demonstrate how predictive maintenance can support maintenance planning.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Loading, cleaning, and processing data |
| Scikit-learn | Training and evaluating ML models |
| Logistic Regression | Classification algorithm |
| Random Forest | Tree-based ensemble classification algorithm |
| Streamlit | Interactive web application |
| Matplotlib | Data visualization |
| Joblib | Saving and loading the trained model |
| JSON | Storing model evaluation metrics |

## 4. Dataset

This project uses the **AI4I 2020 Predictive Maintenance Dataset**, available from the UCI Machine Learning Repository.

Dataset: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

The dataset contains 10,000 synthetic machine records with operational measurements and machine failure labels.

### Important Features

| Feature | Description |
|---|---|
| Type | Machine product quality/type category |
| Air temperature | Ambient air temperature |
| Process temperature | Machine process temperature |
| Rotational speed | Machine rotational speed |
| Torque | Torque measurement |
| Tool wear | Tool wear measurement |
| Machine failure | Target label indicating failure or no failure |

The original dataset also includes identifiers and failure-mode columns. These are excluded from the model's input features to avoid using identifiers or information that would reveal the target during prediction.

The `Machine failure` column is used as the target during model training, not as an input feature.

**Important:** The dataset is synthetic, so the model demonstrates the predictive maintenance workflow rather than guaranteeing performance on real industrial machines.

## 5. How the Project Works

The project follows these steps:

1. **Download the dataset:** `download_data.py` retrieves the dataset and saves it as a CSV file.
2. **Preprocess the data:** `train_model.py` selects the input features and target label and excludes unnecessary or leakage-prone columns.
3. **Split the dataset:** The data is divided into 80% training data and 20% testing data, using stratification to preserve the target-class distribution.
4. **Train the models:** Logistic Regression and Random Forest are trained on the training data.
5. **Evaluate the models:** Both models are evaluated on the held-out test set.
6. **Select the best model:** The model with the higher F1-score is selected.
7. **Save the results:** The selected model is saved to `models/model.joblib`, and evaluation metrics are saved to `models/metrics.json`.
8. **Run the application:** `app.py` loads the saved model and displays the interactive Streamlit interface.
9. **Predict machine failure:** Users enter sensor readings, and the saved model predicts Failure or No Failure.

### Model Selection

The project compares two algorithms:

**Logistic Regression**
- A classification algorithm that learns relationships between input features and the target.
- Provides a relatively simple baseline for comparison.

**Random Forest**
- An ensemble algorithm that combines multiple decision trees.
- Can learn more complex, non-linear patterns in the data.

The models are compared using evaluation metrics, including F1-score. The model with the higher F1-score is selected automatically during training.

Only the selected model is saved in `models/model.joblib` and used for predictions. The application does not combine both models into a single prediction.

## 6. Model Evaluation Metrics

The project evaluates model performance using the following metrics:

| Metric | Meaning |
|---|---|
| Accuracy | Percentage of correct predictions |
| Precision | Among predicted failures, the proportion that are actual failures |
| Recall | Among actual failures, the proportion correctly identified |
| F1-score | Balance between precision and recall |
| ROC-AUC | Measures how well the model distinguishes the two classes across classification thresholds |

Machine failure datasets can have many more normal records than failure records. Therefore, accuracy alone may be misleading. Recall and F1-score are also important when evaluating failure detection.

Actual performance values are generated by `train_model.py` and stored in `models/metrics.json`. Results should be taken from that file rather than assumed in advance.

## 7. Application Features

### Dashboard
- Displays an overview of the dataset.
- Shows machine failure distribution.
- Presents sensor-related charts and summary information.

### Predict Failure
- Allows users to enter machine sensor readings.
- Accepts machine type and operational measurements.
- Predicts Failure or No Failure using the selected model.
- Displays a model score and submitted sensor readings.

The displayed score should not automatically be interpreted as a calibrated real-world probability of failure.

### Model Performance
- Displays evaluation metrics for the trained models.
- Helps compare Logistic Regression and Random Forest.
- Supports understanding why a model was selected.

### Dataset Explorer
- Displays dataset records.
- Shows descriptive statistics.
- Allows users to explore the data used by the project.

## 8. Project Structure

```text
predictive-maintenance/
│
├── .venv/
│   └── Python virtual environment
│
├── data/
│   └── ai4i2020.csv
│
├── models/
│   ├── model.joblib
│   └── metrics.json
│
├── app.py
├── download_data.py
├── train_model.py
├── requirements.txt
└── README.md
```

### File Descriptions

| File | Purpose |
|---|---|
| `app.py` | Streamlit web application and prediction interface |
| `download_data.py` | Downloads and saves the dataset |
| `train_model.py` | Preprocesses data, trains models, evaluates them, and saves the selected model |
| `requirements.txt` | Lists the Python packages required by the project |
| `data/ai4i2020.csv` | Downloaded dataset |
| `models/model.joblib` | Saved selected model |
| `models/metrics.json` | Saved evaluation metrics |
| `.venv/` | Local Python virtual environment |

The `.venv/` folder is local to the development environment and generally should not be uploaded to GitHub.

## 9. Requirements

- Python installed on the computer.
- pip package manager.
- Internet connection for downloading the dataset initially.
- A terminal such as PowerShell or Command Prompt.
- A web browser to access the Streamlit application.

## 10. Installation and Setup

### Step 1: Open the Project Folder

Open PowerShell and run:

```powershell
cd C:\Users\pranj\predictive-maintenance
```

### Step 2: Activate the Virtual Environment

If the `.venv` folder already exists:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, you can use Command Prompt and activate it with:

```cmd
.venv\Scripts\activate.bat
```

### Step 3: Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

## 11. Running the Project

Run the following commands in the project folder, with the virtual environment activated.

### Step 1: Download the Dataset

```powershell
python download_data.py
```

This creates or updates:

```text
data/ai4i2020.csv
```

Run this step when the dataset needs to be downloaded or regenerated.

### Step 2: Train and Evaluate the Models

```powershell
python train_model.py
```

This step trains both algorithms, evaluates their performance, selects the model with the higher F1-score, and saves the model and metrics.

Expected output files:

```text
models/model.joblib
models/metrics.json
```

Run training again whenever you want to retrain the models using the current dataset and training configuration.

### Step 3: Start the Web Application

```powershell
streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open that URL in your browser if it does not open automatically.

You can now explore the dashboard, enter sensor readings, view predictions, compare model metrics, and inspect the dataset.

## 12. Example Prediction Workflow

1. Open the Streamlit application.
2. Navigate to **Predict Failure**.
3. Select a machine type.
4. Enter the air temperature.
5. Enter the process temperature.
6. Enter the rotational speed.
7. Enter the torque.
8. Enter the tool wear value.
9. Submit the readings.
10. Review the predicted class and displayed model score.

The prediction depends on the trained model and the values supplied by the user.

## 13. Current Project Scope

### Implemented

- Dataset download and CSV storage.
- Data preprocessing and feature selection.
- Training of Logistic Regression and Random Forest.
- Evaluation using classification metrics.
- Automatic model selection based on F1-score.
- Saving and loading the selected model.
- Streamlit dashboard.
- Interactive machine failure prediction.
- Model performance page.
- Dataset exploration.

### Current Limitations

- Uses a synthetic dataset rather than live industrial sensor readings.
- Does not connect to IoT devices or physical machinery.
- Does not provide remaining useful life (RUL) predictions.
- Does not automatically send maintenance alerts.
- Predictions depend on the dataset and training process and are not guaranteed to generalize to real machines.

## 14. Future Improvements

Possible future enhancements include:

- Connecting real-time sensor data through IoT devices.
- Adding maintenance alerts when risk thresholds are exceeded.
- Testing on real industrial datasets.
- Improving recall for machine failure detection.
- Adding model explainability to show which features influence predictions.
- Deploying the application online.

These are potential improvements and are not currently implemented features.

## 15. Learning Outcomes

Through this project, the following concepts are demonstrated:

- Data preprocessing with Pandas.
- Supervised machine learning classification.
- Training and testing data separation.
- Model evaluation and comparison.
- Automatic selection of a model using a chosen metric.
- Saving and loading models with Joblib.
- Building an interactive ML application using Streamlit.

## 16. Conclusion

Machine Failure Prediction Using Machine Learning demonstrates how historical machine sensor data can be used to build a classification model for predictive maintenance.

By training and comparing Logistic Regression and Random Forest, the project automatically selects a model based on F1-score and uses it in an interactive Streamlit application.

The project provides a practical introduction to machine learning workflows and demonstrates the potential of predictive analytics to support industrial maintenance decisions.

## 17. Dataset Reference

AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

---

**Project:** Machine Failure Prediction Using Machine Learning  
**Language:** Python  
**Machine Learning:** Scikit-learn  
**Interface:** Streamlit  
**Purpose:** Educational predictive maintenance prototype

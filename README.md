# `Financial Fraud Detection System

This project implements a comparative study of several machine learning models for detecting fraudulent credit card transactions. It includes data preprocessing, model training, evaluation, calibration analysis, and visualization on a real-world imbalanced dataset.

## Project Structure

```
├── EDA.ipynb
├── README.md
├── config
│   └── config.yaml
├── datasets
│   └── creditcard.csv
├── main.py
├── requirements.txt
├── results
└── utils
    ├── data_loader.py
    ├── evaluate_model.py
    ├── preprocess.py
    └── train_model.py
```



## Dataset

The dataset used is the [Credit Card Fraud Detection dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), consisting of 284,807 transactions with 492 fraud cases (0.17% fraud rate). Features `V1`–`V28` are PCA components, along with `Time`, `Amount`, and `Class` as the target.

## Configuration

All runtime settings can be modified in `config/config.yaml`:

```yaml
dataset: "creditcard.csv"
model: "ANN"   # Options: RandomForest, LogisticRegression, SVM, NaiveBayes, KNN, ANN
test_size: 0.30
random_state: 42
shap_sample_size: 1000
```

## How to run

1. Install dependencies:

```
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

```

2. Download the dataset and put it in the ``datasets\`` folder.
3. Run the pipeline:

```
python3 main.py
```

All results (trained models, evaluation plots, confusion matrices, ROC/PR/Calibration curves) will be saved in the `results/` directory.

## Models Supported

* Logistic Regression
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Naive Bayes
* Random Forest
* Neural Network (ANN)

## Evaluation Metrics

* Precision, Recall, F1, F2 Score, Accuracy
* ROC Curve and AUC-ROC
* Precision-Recall Curve and AUPRC
* Calibration Curves (Reliability Diagrams)


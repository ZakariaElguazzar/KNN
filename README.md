# K-Nearest Neighbors (KNN) Classifier

## Overview

This Jupyter Notebook (`knn.ipynb`) implements the **K-Nearest Neighbors (KNN)** classification algorithm using **scikit-learn**. The notebook demonstrates data preprocessing, model training, hyperparameter tuning, and evaluation of a KNN classifier on the **Breast Cancer dataset**.

## Features

- Loads the **Breast Cancer dataset** from `scikit-learn`.
- Splits data into training and testing sets.
- Normalizes features using **scikit-learn's Normalizer**.
- Trains a **KNN classifier** using different distance metrics (e.g., Manhattan).
- Evaluates model performance using **accuracy score, confusion matrix, and cross-validation**.

## Requirements

To run the notebook, install the following dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## Usage

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook knn.ipynb
   ```
2. Run all the cells sequentially.
3. Modify hyperparameters (e.g., `n_neighbors`, `metric`) and re-run the model to observe performance changes.

## Dataset

This notebook uses the **Breast Cancer dataset**, which is a binary classification dataset containing:

- **569 samples**
- **30 numerical features**
- **2 classes**: malignant (cancerous) vs. benign (non-cancerous)

## Model Details

The KNN classifier is implemented using `sklearn.neighbors.KNeighborsClassifier`, with key parameters:

- **Number of neighbors (**``**)**: Chosen based on cross-validation.
- **Distance metric (**``**)**: Uses **Manhattan distance**.
- **Data normalization**: Features are normalized using `Normalizer()` to improve distance-based classification performance.

## Results

The notebook evaluates model performance using:

- **Accuracy Score**: Measures overall model performance.
- **Confusion Matrix**: Visualizes correct and incorrect predictions.
- **Cross-Validation**: Ensures robustness by evaluating different train-test splits.

## Author

This project was completed as part of **LAB 2** by **Zakaria Elguazzar**.

## License

This project is open-source and available for educational purposes.


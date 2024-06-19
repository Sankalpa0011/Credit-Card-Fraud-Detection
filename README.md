---

# Credit Card Fraud Detection

This repository contains a Jupyter Notebook for detecting credit card fraud using machine learning techniques.

## Project Overview

The objective of this project is to build a machine learning model that can detect fraudulent credit card transactions. The dataset used is highly imbalanced, with a very small proportion of fraudulent transactions. Various data preprocessing techniques, feature selection methods, and classification algorithms have been applied to build an effective fraud detection model.

## Dataset

The dataset used in this project is from the Kaggle competition: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

- **Number of Instances**: 284,807
- **Number of Features**: 30
- **Class Distribution**: 0 - 99.83% (Non-fraud), 1 - 0.17% (Fraud)

## Requirements

The following Python libraries are required to run the notebook:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn

You can install the required libraries using the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

## Notebook Contents

1. **Introduction**: Overview of the problem and dataset.
2. **Data Exploration**: Loading and exploring the dataset.
3. **Data Preprocessing**: Handling missing values, feature scaling, and dealing with imbalanced data.
4. **Feature Engineering**: Creating new features and selecting important features.
5. **Model Building**: Training various machine learning models and evaluating their performance.
6. **Model Evaluation**: Comparing models using various metrics and selecting the best model.
7. **Conclusion**: Summary of findings and future work.

## How to Use

1. Clone this repository:

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
```

2. Navigate to the project directory:

```bash
cd credit-card-fraud-detection
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Run the Jupyter Notebook:

```bash
jupyter notebook Credit\ Card\ Fraud\ Detection.ipynb
```

## Results

### Model Performance

- **Training Accuracy**: 0.9416
- **Testing Accuracy**: 0.9492

### Classification Reports

**Training Classification Report**:
```
               precision    recall  f1-score   support

           0       0.92      0.96      0.94       393
           1       0.96      0.92      0.94       394

    accuracy                           0.94       787
   macro avg       0.94      0.94      0.94       787
weighted avg       0.94      0.94      0.94       787
```

**Testing Classification Report**:
```
               precision    recall  f1-score   support

           0       0.95      0.95      0.95        99
           1       0.95      0.95      0.95        98

    accuracy                           0.95       197
   macro avg       0.95      0.95      0.95       197
weighted avg       0.95      0.95      0.95       197
```

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. For any issues, please open a new issue in the GitHub repository.

## Acknowledgements

- Kaggle for providing the dataset.
- The open-source community for the libraries and tools used in this project.

---

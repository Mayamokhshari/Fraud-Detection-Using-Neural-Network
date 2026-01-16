# Credit Card Fraud Detection using Neural Networks
Overview

In this project, we leverage the power of neural networks to detect potentially fraudulent credit card transactions. The dataset in use has been sourced from Kaggle's Credit Card Fraud Detection challenge.
- Dataset

    The dataset comprises transactions made via credit cards.
    From a total of 284,807 transactions, only 492 were flagged as fraud.
    This imbalance, with the fraudulent class accounting for a mere 0.172% of the dataset, posed a challenge.

- Approach

    Data Preprocessing:
        Addressed the data imbalance by applying the Synthetic Minority Over-sampling Technique (SMOTE) to oversample the minority class.

    Neural Network Model:
        Developed using the powerful PyTorch library.

    Evaluation Metrics:
        Beyond mere accuracy, which can be misleading with imbalanced datasets, we also monitored Precision, Recall, F1 Score, and ROC AUC Score to holistically evaluate the model's performance.

- Results

    Accuracy: 98.16%
    Precision: 7.75%
    Recall: 88.78%
    F1 Score: 14.26%
    ROC AUC Score: 93.38%

- Analysis

    Accuracy: Though high, accuracy alone doesn't tell the whole story especially with imbalanced datasets.
    Recall: The standout metric with a high score of 88.78%, showing the model's strength in identifying most of the fraudulent transactions.
    Precision: At 7.75%, it's an area where the model can improve, indicating a higher number of false positives.

- Further Steps and Improvements

    Engage in hyperparameter tuning for the model.
    Experiment with feature engineering to enhance input data.
    Evaluate the potential of anomaly detection models for this use case.
    Consider ensemble methods to potentially boost performance.

- Conclusion

Fraud detection, given its complexity and the stakes involved, is a challenging domain. The results achieved here are promising and point towards a strong baseline model. Adjustments and refinements based on specific real-world use cases and costs associated with false positives/negatives would further enhance its utility.


# Fraud Detection using LSTM on the PaySim Dataset

Detecting fraudulent transactions with the mighty power of Long Short-Term Memory (LSTM) neural networks.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Development](#model-development)
   - [Baseline Model](#baseline-model)
   - [Improvements & Iterations](#improvements--iterations)
5. [Results & Discussions](#results--discussions)
6. [Conclusion & Future Work](#conclusion--future-work)

## Introduction

The world of finance is filled with transactions—most genuine, some not. Our mission? Using LSTMs to shine a light on those sneaky fraudulent ones hiding within the PaySim dataset.

## Getting Started

**Prerequisites**:
- Python 3.x
- Libraries: `pandas`, `keras`, `imbalanced-learn`

```bash
pip install pandas keras imbalanced-learn

Data Preprocessing
Loading and Cleaning

    Loaded the PaySim dataset using pandas.
    Dropped columns like nameOrig and nameDest that weren't contributing much.
    Encoded categorical variables using pd.get_dummies() to make them digestible for our neural network.

Scaling and Splitting

    Normalized numerical columns with MinMaxScaler so they're all between 0 and 1, ensuring uniformity for our LSTM.
    Split our data into training and testing sets (80-20 split) to later evaluate our model.

Model Development
Baseline Model

    Used a simple LSTM architecture to establish a baseline performance.
    Added a Dropout layer to prevent overfitting.
    Introduced a Dense layer with sigmoid activation to classify transactions as genuine or fraudulent.

Improvements & Iterations
Addressing Class Imbalance with SMOTE

    Used SMOTE to balance our classes, adding synthetic samples to the minority fraudulent class.
    Reshaped data to fit the 3D input structure LSTMs love so much.

Enhanced Model Iteration

    Stacked more LSTM layers for deeper learning.
    Introduced Bidirectional LSTM layers to learn from both past and future data.
    Added additional Dropout layers to further prevent overfitting.

Results & Discussions

Initially, our detective (the model) was a bit of a rookie, catching fraudulent transactions with a precision and recall of:

    Precision: 83% for fraudulent class
    Recall: 71% for fraudulent class

After training, our detective got some cool gadgets (SMOTE, stacked LSTMs) and became a seasoned pro:

    Precision: 100% for fraudulent class
    Recall: 79% for fraudulent class

This meant our model was almost perfect in spotting frauds when it claimed to and caught a significant majority of the total frauds present.
Conclusion & Future Work

Our journey took our detective from being a rookie to a top agent in spotting fraudulent transactions! Future plans include:

    Exploring Feature Engineering: Can we extract more insights from our data?
    Experimenting with Ensemble Methods: Combine the might of multiple models!
    Hyperparameter Tuning: Every top agent can still fine-tune their skills.

Thanks for joining this adventure! Here's to making the financial world a safer place. 

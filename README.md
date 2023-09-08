# Persian Vowels Recognition

## Overview
This project is aimed at recognizing Persian vowels (Ezafe), which is a grammatical construction in the Persian language. The goal is to build a machine learning model that can predict whether a word should take vowel (Ezafe) or not based on its surrounding context and part of speech.

## Dependencies
This project uses the following Python libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- xgboost

## Data
The input data is read from the "updated_bijankhan_corpus.csv" file. It contains information about the part of speech, Ezafe tags, and other linguistic features.

## Data Preprocessing
- The part of speech tags is transformed into numerical values using label encoding.
- Additional features, such as the next and last part of speech tags, are added to the dataset.

## Classification Models
This project uses various machine learning classification models to predict vowel (Ezafe) tags:
- Random Forest
- Decision Tree
- Naive Bayes
- Logistic Regression
- Multilayer Perceptron (MLP)
- Gradient Boosting Classifier
- XGBoost

Each model is evaluated for its accuracy, precision, and recall.

## Feature Engineering
To improve model performance, additional features are introduced:
- Dependency on the last two words
- Dependency on the next two words
- The last character of the word
- The last punctuation mark of the word

These features are incorporated into the dataset and used to train and evaluate the models.

## Running the Code
You can run the code by executing the Python script provided. Make sure to have the required dependencies installed.

## Results
The project evaluates the performance of each model with different sets of features and provides accuracy, precision, and recall scores for each model. The best model and feature combination can be determined based on the results.

## Conclusion
This project demonstrates the effectiveness of machine learning models in recognizing Persian vowel tags. The addition of various linguistic features and thorough evaluation helps improve the accuracy of the models.

# Decision Tree Classifier for Bank Marketing Data

A professional Python implementation of a Decision Tree Classifier for analyzing the Bank Marketing dataset, utilizing Pandas for data handling and scikit-learn for model training and evaluation.

## Project Overview
This project implements a Decision Tree Classifier to predict customer responses in the Bank Marketing dataset, which contains data related to direct marketing campaigns of a banking institution. The classifier is built using scikit-learn, with data preprocessing and analysis performed using Pandas. The project is designed for data scientists and machine learning practitioners seeking to model customer behavior and evaluate predictive performance.

## Key Features
- **Data Loading and Preprocessing**: Efficiently handles the Bank Marketing dataset with one-hot encoding for categorical variables.
- **Model Training**: Trains a Decision Tree Classifier with configurable parameters.
- **Model Evaluation**: Provides comprehensive metrics including accuracy, confusion matrix, and classification report.
- **Scalable Pipeline**: Easily adaptable for other classification datasets or machine learning models.
- **Reproducible Results**: Ensures consistency with a fixed random seed.

## Dataset
The Bank Marketing dataset is sourced from a banking institution's direct marketing campaigns. It includes features such as customer demographics, campaign details, and subscription outcomes. The dataset (`bank-full.csv`) is expected to be in CSV format with a semicolon (`;`) delimiter.

## System Requirements
- **Python**: Version 3.8 or higher
- **Libraries**:
  - `pandas` for data manipulation
  - `scikit-learn` for machine learning
- **Operating System**: Windows, macOS, or Linux

## Installation Guide
1. **Install Dependencies**:
   ```bash
   pip install pandas scikit-learn
   ```

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-organization/bank-marketing-decision-tree.git
   cd bank-marketing-decision-tree
   ```

3. **Prepare the Dataset**:
   Place the `bank-full.csv` file in the project directory or update the file path in `decision_tree_classifier.py`.

## Usage Instructions
1. **Configure the Dataset Path**:
   Update the `url` variable in `decision_tree_classifier.py` to point to the dataset:
   ```python
   url = "path/to/bank-full.csv"
   ```

2. **Run the Script**:
   Execute the analysis from the command line:
   ```bash
   python decision_tree_classifier.py
   ```

## Technical Workflow
The script follows a structured machine learning pipeline:

1. **Data Loading**:
   Loads the Bank Marketing dataset into a Pandas DataFrame using a semicolon delimiter.

2. **Data Preprocessing**:
   - Drops the `duration` column to avoid bias in predictions.
   - Applies one-hot encoding to categorical columns (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`, `y`) using `pd.get_dummies`.

3. **Data Splitting**:
   Splits the dataset into training (80%) and testing (20%) sets with a fixed random seed for reproducibility.

4. **Model Training**:
   Instantiates and trains a `DecisionTreeClassifier` from scikit-learn on the training data.

5. **Model Evaluation**:
   Evaluates the model on the test set, computing:
   - Accuracy score
   - Confusion matrix
   - Classification report (precision, recall, F1-score)

## Outputs and Evaluation Metrics
- **Console Outputs**:
  - Accuracy score of the classifier on the test set.
  - Confusion matrix detailing true positives, false positives, true negatives, and false negatives.
  - Classification report providing precision, recall, F1-score, and support for each class.
- **Potential Extensions**:
  - Visualizations of the decision tree or feature importance can be added.
  - Model performance metrics can be saved as CSV or JSON files.

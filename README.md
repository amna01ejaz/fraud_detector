```markdown
# Credit Card Fraud Detection

This project handles the implementation of a machine learning workflow designed to identify fraudulent credit card transactions. The primary objective is to address extreme class imbalance using synthetic sampling techniques and optimize evaluation based on classification metrics rather than raw accuracy.

## Dataset
The model trains on the Credit Card Fraud Detection dataset provided by ULB on Kaggle. It contains transactions made by European cardholders, where only 0.172% of the operations are flagged as fraudulent.

## Project Structure
* app.py: The main executable Python file handling data scaling, SMOTE application, random forest training, and metric evaluation.
* creditcard.csv: The highly dimension-reduced numerical dataset containing transaction records.

## Prerequisites and Installation
This project requires libraries for handling imbalanced datasets in addition to standard machine learning packages. Install them using the following command:

```bash
pip install pandas scikit-learn imbalanced-learn
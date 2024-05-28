'''
Churn Library Testing File

author: Abhijith
date: February 13th, 2024
'''

import logging
from pathlib import Path
from datetime import datetime
import churn_library as cls

DATA_FOLDER = "./data"
IMAGES_FOLDER = "./images"
LOGS_FOLDER = "./logs"
EDA_FOLDER = "./images/eda"
REPORTS_FOLDER = "./images/reports"
RESULTS_FOLDER = "./images/results"
FEATURES_FOLDER = "./images/results/feature_imp"
MODELS_FOLDER = "./models"

current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]
FINAL_COLUMNS = ['Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
    'Income_Category_Churn', 'Card_Category_Churn'
]

EDA_COLUMNS = [
    "Churn_histogram",
    "Customer_Age_histogram",
    "Marital_Status_histogram",
    "Total_Trans_Ct_histogram",
    "correlation"
]
def test_import(path):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = cls.import_data(path)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return df

def test_eda(df, columns):
    '''
    test perform eda function
    '''
    model_columns =  ["Churn", "Customer_Age", "Marital_Status", "Total_Trans_Ct"]
    cls.perform_eda(df, model_columns)
    for file in columns:
        file_path = Path().joinpath(f'{EDA_FOLDER}/{file}.png')
        try:
            assert file_path.is_file()
        except AssertionError as err:
            logging.error("ERROR: Eda results not found.")
            raise err
    logging.info("SUCCESS: EDA results were saved!")

def test_encoder_helper(df, columns):
    '''
    test encoder helper
    '''
    data = cls.encoder_helper(df, columns, "Churn")

    #Categorical columns exists test
    try:
        for col in columns:
            assert col in df.columns
    except AssertionError as err:
        logging.error("ERROR: Missing categorical columns")
        raise err
    logging.info("SUCCESS: Categorical columns correctly encoded.")
    return data

def test_perform_feature_engineering(encoded_df):
    '''
    Splits the data into training and testing sets and performs
    necessary checks to ensure consistency in the splits.

    Args:
        data_frame (DataFrame): The encoded dataset.
        classifier (Classifier): An instance of the classifier.

    Returns:
        tuple: A tuple containing the split data (X_train, X_test, y_train, y_test).
    '''
    # Perform feature engineering
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        encoded_df, FINAL_COLUMNS, "Churn")

    # Ensure consistency in the splits
    try:
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
    except AssertionError:
        logging.error("ERROR: Train and Test datasets do not match")
        raise
    logging.info("SUCCESS: Train test split correctly.")
    return X_train, X_test, y_train, y_test

def test_train_models(split_data):
    '''
    test train_models
    '''
    x_train, x_test, y_train, y_test = split_data
    cls.train_models(x_train, x_test, y_train, y_test)
    #Test for checking if model file exists
    models = ['logistic_regression_model',"random_forest_classifier_model"]
    for model in models:
        file_path = Path().joinpath(f'{MODELS_FOLDER}/{model}.pkl')
        try:
            assert file_path.is_file()
        except AssertionError as err:
            logging.error("ERROR: Models not found.")
            raise err
    logging.info("SUCCESS: Models were saved!")

if __name__ == "__main__":
    df = test_import(f"{DATA_FOLDER}/bank_data.csv")
    test_eda(df, EDA_COLUMNS)
    encoded_df = test_encoder_helper(df, CAT_COLUMNS)
    featured_data = test_perform_feature_engineering(encoded_df)
    test_train_models(featured_data)
    
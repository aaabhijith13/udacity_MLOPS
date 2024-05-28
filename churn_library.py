'''
Churn Library Independent File

author: Abhijith
date: February 13th, 2024
'''

import os
from datetime import datetime
import logging
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, plot_roc_curve

DATA_FOLDER = "./data"
IMAGES_FOLDER = "./images"
LOGS_FOLDER = "./logs"
EDA_FOLDER = "./images/eda"
REPORTS_FOLDER = "./images/reports"
RESULTS_FOLDER = "./images/results"
FEATURES_FOLDER = "./images/results/feature_imp"
MODEL_FOLDER = "./models"

CAT_COLUMNS = ['Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category']

FINAL_COLUMNS = ['Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
    'Income_Category_Churn', 'Card_Category_Churn']

EDA_COLUMNS = ["Churn", "Customer_Age", "Marital_Status", "Total_Trans_Ct"]

current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
 filename=f'./logs/test_results-{formatted_time}.log',
  level=logging.DEBUG,
  filemode='w',
  format='%(name)s - %(levelname)s - %(message)s')


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    try:
        logging.info("INFO: Path of the file is %s", pth)
        return pd.read_csv(pth)
    except FileNotFoundError:
        logging.error("ERROR: File not found in the destination")
        return None  # Return None or handle the exception appropriately


def perform_eda(data_frame, columns):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe
            columns: list of columns to perform EDA on

    output:
            None
    '''

    # Convert Attrition Flag to a binary variable Churn
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    for col in columns:
        plt.clf()
        plot_size = (20, 10)
        plt.figure(figsize=plot_size)
        data_frame[col].hist()
        plt.savefig(f'{EDA_FOLDER}/{col}_histogram.png')
        logging.info("SUCCESS: Saved %s histogram", col)

    # Plot heatmap of correlation matrix
    plt.clf()
    plt.figure(figsize=(12, 8))
    sns.heatmap(data_frame.corr(), annot=False,
                cmap='Dark2_r', linewidths = 2)
    plt.title('Correlation Heatmap')
    plt.savefig(f'{EDA_FOLDER}/correlation.png')
    logging.info("SUCCESS: Saved correlation heatmap")

def encoder_helper(data_frame, columns, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category -
    associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            columns: list of columns that contain categorical features
            response: string of response name

    output:
            data_frame: pandas dataframe with new columns for
    '''
    logging.info("INFO: Starting categorical encoding")
    try:
        for category in columns:
            category_lst = []
            category_groups = data_frame.groupby(category).mean()[response]
            for val in data_frame[category]:
                category_lst.append(category_groups.loc[val])
            data_frame[f'{category}_Churn'] = category_lst
        logging.info("Categorical encoding completed successfully.")
        return data_frame
    except ValueError as value_issue:
        logging.error("Error occurred during categorical encoding: %s",
                      str(value_issue))
        return None

def perform_feature_engineering(data_frame, column_list, response):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name [optional
              argument that could be used for
              naming variables or index y column]
    output:
              x_train: x training data
              x_test: x testing data
              y_train: y training data
              y_test: y testing data
    '''
    logging.info("Starting feature engineering and train-test split...")
    try:
        # Splitting the data into features (x) and target variable (y)
        x_data_frame = data_frame[column_list]
        y_data_frame = data_frame[response]
        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(
            x_data_frame, y_data_frame, test_size=0.3, random_state=42)
        logging.info('Feature engineering and split')
        return x_train, x_test, y_train, y_test
    except ValueError as value_prob:
        logging.error("Error occurred during featureengineering: %s", str(value_prob))
        return None, None, None, None
def classification_report_image(params):
    '''
    produces classification report for training
    and testing results and stores report as image
    in images folder
    input:
            model_name: model name associated with the model
            y_train: training response values
            y_test:  test response values
            y_train_preds: training predictions from model
            y_test_preds: test predictions from model
    output:
             None
    '''
    model_name, y_train,y_test, y_train_preds,y_test_preds = params
    try:
        if model_name == "Random Forest Classifier":
            plt.clf()
            plt.rc('figure', figsize=(5, 5))
            plt.text(0.01, 1.25, str(model_name),
                     {'fontsize': 10},
                     fontproperties = 'monospace')
            plt.text(0.01, 0.05,
                     str(classification_report(y_test, y_test_preds)),
                     {'fontsize': 10},
                     fontproperties = 'monospace')
            plt.text(0.01, 0.6, str('Random Forest Test'),
                     {'fontsize': 10}, fontproperties = 'monospace')
            plt.text(0.01, 0.7,
                     str(classification_report(y_train, y_train_preds)),
                     {'fontsize': 10},
                     fontproperties = 'monospace')
            plt.axis('off')
            plt.savefig(f'{REPORTS_FOLDER}/{model_name}_{formatted_time}.png')
        else:
            plt.clf()
            plt.rc('figure', figsize=(5, 5))
            plt.text(0.01, 1.25, model_name,
                     {'fontsize': 10}, fontproperties = 'monospace')
            plt.text(0.01, 0.05,
                     str(classification_report(y_train, y_train_preds)),
                     {'fontsize': 10},
                     fontproperties = 'monospace')
            plt.text(0.01, 0.6, str(f'{model_name} Test'),
                     {'fontsize': 10}, fontproperties = 'monospace')
            plt.text(0.01, 0.7,
                     str(classification_report(y_test, y_test_preds)),
                     {'fontsize': 10},
                     fontproperties = 'monospace')
            plt.axis('off')
            plt.savefig(f'{REPORTS_FOLDER}/{model_name}_{formatted_time}.png')
            logging.info("SUCCESS: %s report saved.", model_name)
    except AttributeError as attr_error:
        logging.error("ERROR: Unable to Generate %s", attr_error)


def feature_importance_plot(model, x_data, output_path):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of x values
            output_pth: path to store the figure
    output:
             None
    '''
    try:
                # Calculate feature importances
        importances = model.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        # Rearrange feature names to match sorted features
        names = [x_data.columns[i] for i in indices]

        # Create plot
        plt.clf()
        plt.figure(figsize=(20,5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(x_data.shape[1]), importances[indices])

        # Add feature names as x-axisis labels
        plt.xticks(range(x_data.shape[1]), names, rotation=90)
        plt.savefig(output_path)
        logging.info("SUCCESS: 'Random Forest'_%s Feature Importance saved.",
                     formatted_time)
    except AttributeError:
        logging.error("ERROR: Unable to Generate")

def train_models(x_train, x_test, y_train, y_test):
    '''
    Train models, store model results: images +
    scores, and store models
    input:
              x_train: x training data
              x_test: x testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    logging.info("INFO: Training Started")
    models = [
        (RandomForestClassifier(random_state=42),
         "Random Forest Classifier"),
        (LogisticRegression(solver='lbfgs', max_iter=3000),
         "Logistic Regression")
    ]
    for model, model_name in models:
        try:
            logging.info("INFO: Initiated %s", model_name)
            if model_name == "Random Forest Classifier":
                param_grid = {
                    'n_estimators': [200, 500],
                    'max_features': ['auto', 'sqrt'],
                    'max_depth' : [4,5,100],
                    'criterion' :['gini', 'entropy']
                }
                cross_validator = GridSearchCV(estimator=model,
                                               param_grid=param_grid,
                                               cv=5)
                logging.info("SUCCESS: %s Grid Search completed.",
                             model_name)
                # Fit the GridSearchCV object
                cross_validator.fit(x_train, y_train)
                # Get predictions using the best estimator
                y_train_preds = cross_validator.best_estimator_.predict(x_train)
                y_test_preds = cross_validator.best_estimator_.predict(x_test)
                #Classification report
                params = model_name, y_train, y_test,y_train_preds,y_test_preds
                classification_report_image(params)
            else:
                # Fit the Logistic Regression model
                model.fit(x_train, y_train)
                # Get predictions using the model
                y_train_preds = model.predict(x_train)
                y_test_preds = model.predict(x_test)
                #Classification report
                params = model_name, y_train, y_test,y_train_preds,y_test_preds
                classification_report_image(params)
            # Construct the full paths for the output files
            train_preds_file = os.path.join(RESULTS_FOLDER,
            f"train_predictions_{model_name}_{formatted_time}.csv")
            test_preds_file = os.path.join(RESULTS_FOLDER,
            f"test_predictions_{model_name}_{formatted_time}.csv")
            # Save the predictions to the specified folder
            pd.DataFrame(y_train_preds).to_csv(train_preds_file,
                                               header=False, index=False)
            pd.DataFrame(y_test_preds).to_csv(test_preds_file,
                                              header=False, index=False)
            # Log the information
            logging.info("SUCCESS: Train predictions saved to %s.",
                         train_preds_file)
            logging.info("SUCCESS: Test predictions saved to %s.",
                         test_preds_file)
            # Plot ROC curve
            plt.clf()
            plt.figure(figsize=(15, 8))
            axis = plt.gca()
            # Plot using the best estimator if Random Forest Classifier
            if model_name == "Random Forest Classifier":
                plot_roc_curve(cross_validator.best_estimator_,
                               x_test,
                               y_test,
                               ax=axis,
                               alpha=0.8)
                plt.savefig(f'{RESULTS_FOLDER}/ROC_Curve_{model_name}__{formatted_time}.png')
                logging.info("INFO: ROC curve plotted for %s__%s.",
                             model_name, formatted_time)
                #Feature Importance
                feature_importance_plot(cross_validator.best_estimator_,                 x_test,f'{FEATURES_FOLDER}/feature_importance_{model_name}_{formatted_time}.png')
                logging.info("SUCCESS: Feature importance plotted for %s__%s.",
                             model_name, formatted_time)
            else:
                plot_roc_curve(model, x_test, y_test,
                               ax=axis, alpha=0.8)
                plt.savefig(
                f'{RESULTS_FOLDER}/ROC_Curve_{model_name}_{formatted_time}.png')
                logging.info("INFO: ROC curve plotted for %s_%s.",
                             model_name, formatted_time)
            # Save best model
            joblib.dump(model,
                        f'{MODEL_FOLDER}/{model_name.lower().replace(" ", "_")}_model.pkl')
            logging.info("SUCCESS: Best %s model saved.", model_name)
        except ValueError as value_error:
            logging.error("ValueError occurred: %s", str(value_error))
    logging.info("SUCCESS: Training process completed.")
    
if __name__ == "__main__":
    first_data = import_data(f"{DATA_FOLDER}/bank_data.csv")#import data
    perform_eda(first_data, EDA_COLUMNS)#Perform EDA
    #Data Engineering
    encoded_data_frame = encoder_helper(first_data, CAT_COLUMNS, "Churn")
    x_train_data, x_test_data,y_train_data, y_test_data = perform_feature_engineering(
        encoded_data_frame, FINAL_COLUMNS, "Churn") #Feature Engineering
    train_models(x_train_data,
                 x_test_data,
                 y_train_data,
                 y_test_data) #Model Training

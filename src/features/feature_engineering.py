import numpy as np
import pandas as pd
import os 
from src.logger import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(data_path):
    try:
        df=pd.read_csv(data_path)
        logging.info('preprocessed data loaded for feature engineering...')
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise
def new_features(df):
    try:
        df["orgBalanceDiff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
        df["destBalanceDiff"] = df["newbalanceDest"] - df["oldbalanceDest"]
        logging.info('new features created')

        return df
    except Exception as e:
        logging.error('unexpected error occured %s',e)
def label_encoding(df):
    try:
        le = LabelEncoder()
        df["type"] = le.fit_transform(df["type"])
        logging.info('label encoding done')
        return df
    except Exception as e:
        logging.error(f"Error in label encoding: {e}")
        raise

def standard_scaling(df):
    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X)
    # X_test_scaled = scaler.transform(X_test)
    logging.info('standard_scaling done')

    final_df = pd.DataFrame(X_train_scaled,columns=X.columns)
    final_df['isFraud'] = y.values

    return final_df

def save_data(df,file_path):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info('Data saved to %s', file_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
    
        raise
def main():
    try:
        train_data = load_data('./datas/interim/train_processed.csv')
        test_data = load_data('./datas/interim/test_processed.csv')

        train_data=new_features(train_data)
        test_data=new_features(test_data)

        train_data=label_encoding(train_data)
        test_data=label_encoding(test_data)

        train_data=standard_scaling(train_data)
        test_data=standard_scaling(test_data)

        save_data(train_data,os.path.join("./datas", "processed", "train_scaled.csv"))
        save_data(test_data,os.path.join("./datas", "processed", "test_scaled.csv"))
        
    except Exception as e:
        logging.error('unexpected error occured in feature engineering %s',e)
        raise

if __name__=='__main__':
    main()
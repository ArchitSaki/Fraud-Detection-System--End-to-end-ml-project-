import numpy as np
import pandas as pd
import os 
from src.logger import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

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

def train_model(X_train,y_train):
    try:
        rf=RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200)
        rf.fit(X_train,y_train)
        logging.info('model training started.....')
        return rf
    except Exception as e:
        logging.error('Unexpected error occurred while training the model: %s', e)
        raise
def save_model(model,file_path):
    try:
        with open(file_path,'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise
def main():
    try:
        train_data=load_data("./datas/processed/train_scaled.csv")
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train)
        
        save_model(clf, 'models/model.pkl')
    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
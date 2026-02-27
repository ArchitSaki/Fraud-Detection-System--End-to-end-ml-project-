import numpy as np
import pandas as pd
import os
from src.logger import logging


def preprocessor(df):
    logging.info('preprocessing started...')
    if df.isnull().sum().sum()>0:
        logging.info('null values found in df')
        df=df.dropna()
        logging.info('null values removed from df')
    if df.duplicated().sum()>0:
        logging.info('duplicate values found in df')
        df=df.drop_duplicates()
        logging.info('duplicate values removed from df')
    return df
def main():
    try:
        train_data=pd.read_csv('./datas/raw/train.csv')
        test_data = pd.read_csv('./datas/raw/test.csv')
        logging.info('data loaded properly')
        train_processed_data = preprocessor(train_data)
        test_processed_data = preprocessor(test_data)

        # Store the data inside data/processed
        data_path = os.path.join("./datas", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logging.info('Processed data saved to %s', data_path)
    except Exception as e:
        logging.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
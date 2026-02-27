import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from src.logger import logging
# import logging
from src.connections import s3_connection
from dotenv import load_dotenv
import os
import yaml

load_dotenv()

def load_params(params_path):
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise
def load_data(data_url):
    try:
        df = pd.read_csv(data_url)
        logging.info('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise


def format_data(df):
    try:
        logging.info('Table formatting started')
        df = df.drop(["nameOrig", "nameDest", "isFlaggedFraud"], axis=1)
        return df
    except Exception as e:
        logging.error('Unexpected error occurred while formatting the data: %s', e)
        raise


def save_data(train_data, test_data, data_path):
    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logging.info('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        data_url = "https://raw.githubusercontent.com/ArchitSaki/Fraud-Detection-System--End-to-end-ml-project-/main/notebooks/data.csv"

        # df = load_data(data_url)
        s3 = s3_connection.S3_operations(
                bucket_name=os.getenv("AWS_BUCKET_NAME"),
                aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY")
                )
        df=s3.get_file_s3("data.csv")
        df = format_data(df)

        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        
        save_data(train_data, test_data, "datas")

    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
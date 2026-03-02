import numpy as np
import pandas as pd
import os
import pickle
from src.logger import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(data_path):
    try:
        df = pd.read_csv(data_path)
        logging.info('Preprocessed data loaded for feature engineering...')
        return df
    except Exception as e:
        logging.error(f'Error loading data: {e}')
        raise


def new_features(df):
    try:
        df["orgBalanceDiff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
        df["destBalanceDiff"] = df["newbalanceDest"] - df["oldbalanceDest"]
        logging.info('New features created')
        return df
    except Exception as e:
        logging.error(f'Error in feature creation: {e}')
        raise


# -------------------------
# LABEL ENCODING
# -------------------------
def label_encoding_fit(df):
    try:
        le = LabelEncoder()
        df["type"] = le.fit_transform(df["type"])
        logging.info('Label encoding fitted on training data')
        return df, le
    except Exception as e:
        logging.error(f"Error in label encoding: {e}")
        raise


def label_encoding_transform(df, le):
    df["type"] = le.transform(df["type"])
    return df


# -------------------------
# SCALING
# -------------------------
def standard_scaling_fit(df):
    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    final_df = pd.DataFrame(X_scaled, columns=X.columns)
    final_df["isFraud"] = y.values

    logging.info("Scaler fitted on training data")
    return final_df, scaler


def standard_scaling_transform(df, scaler):
    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]

    X_scaled = scaler.transform(X)

    final_df = pd.DataFrame(X_scaled, columns=X.columns)
    final_df["isFraud"] = y.values

    return final_df


# -------------------------
# SAVE MODEL
# -------------------------
def save_model(model, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
    logging.info(f'Model saved to {file_path}')


def save_data(df, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    logging.info(f'Data saved to {file_path}')


# -------------------------
# MAIN PIPELINE
# -------------------------
def main():
    try:
        train_data = load_data('./datas/interim/train_processed.csv')
        test_data = load_data('./datas/interim/test_processed.csv')

        # Feature engineering
        train_data = new_features(train_data)
        test_data = new_features(test_data)

        # Label encoding
        train_data, le = label_encoding_fit(train_data)
        test_data = label_encoding_transform(test_data, le)

        # Scaling
        train_data, scaler = standard_scaling_fit(train_data)
        test_data = standard_scaling_transform(test_data, scaler)

        # Save processed data
        save_data(train_data, "./datas/processed/train_scaled.csv")
        save_data(test_data, "./datas/processed/test_scaled.csv")

        # Save artifacts
        save_model(le, "models/label_encoder.pkl")
        save_model(scaler, "models/scaler.pkl")

        logging.info("Feature engineering pipeline completed successfully.")

    except Exception as e:
        logging.error(f'Unexpected error in feature engineering: {e}')
        raise


if __name__ == '__main__':
    main()
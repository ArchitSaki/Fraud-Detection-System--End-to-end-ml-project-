# load test + signature test + performance test

import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "ArchitSaki"
        repo_name = "Fraud-Detection-System--End-to-end-ml-project-"

        # MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load model from MLflow
        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # Load preprocessing objects
        cls.label_encoder = pickle.load(open('models/label_encoder.pkl', 'rb'))
        cls.scaler = pickle.load(open('models/scaler.pkl', 'rb'))

        # Load holdout test data
        cls.holdout_data = pd.read_csv('datas/processed/test_processed.csv')

        # Split features and target
        cls.X_test = cls.holdout_data.drop("isFraud", axis=1)
        cls.y_test = cls.holdout_data["isFraud"]

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None

    # --------------------------------------------------
    # 1. Model Load Test
    # --------------------------------------------------
    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    # --------------------------------------------------
    # 2. Model Signature Test
    # --------------------------------------------------
    def test_model_signature(self):
        # Take one sample
        sample_input = self.X_test.iloc[:1]

        # Apply preprocessing
        sample_scaled = self.scaler.transform(sample_input)

        input_df = pd.DataFrame(
            sample_scaled,
            columns=self.X_test.columns
        )

        # Prediction
        prediction = self.new_model.predict(input_df)

        # Input shape validation
        self.assertEqual(input_df.shape[1], self.X_test.shape[1])

        # Output validation
        self.assertEqual(len(prediction), 1)
        self.assertEqual(len(prediction.shape), 1)

    # --------------------------------------------------
    # 3. Model Performance Test
    # --------------------------------------------------
    def test_model_performance(self):
        # Scale features
        X_scaled = self.scaler.transform(self.X_test)

        X_scaled = pd.DataFrame(X_scaled, columns=self.X_test.columns)

        # Prediction
        y_pred_new = self.new_model.predict(X_scaled)

        # Metrics
        accuracy_new = accuracy_score(self.y_test, y_pred_new)
        precision_new = precision_score(self.y_test, y_pred_new)
        recall_new = recall_score(self.y_test, y_pred_new)
        f1_new = f1_score(self.y_test, y_pred_new)

        # Thresholds
        expected_accuracy = 0.80
        expected_precision = 0.60
        expected_recall = 0.60
        expected_f1 = 0.60

        self.assertGreaterEqual(accuracy_new, expected_accuracy)
        self.assertGreaterEqual(precision_new, expected_precision)
        self.assertGreaterEqual(recall_new, expected_recall)
        self.assertGreaterEqual(f1_new, expected_f1)


if __name__ == "__main__":
    unittest.main()
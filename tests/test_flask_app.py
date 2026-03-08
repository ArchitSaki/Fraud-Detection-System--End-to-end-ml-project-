import unittest
import os

# Ensure env variable exists during tests
os.environ["CAPSTONE_TEST"] = "test_token"

from flask_app.app import app

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Fraud Detection System', response.data)

    def test_predict_page(self):
        test_data = {
            "type": "PAYMENT",
            "amount": "1000",
            "nameOrig": "C123",
            "oldbalanceOrg": "5000",
            "newbalanceOrig": "4000",
            "nameDest": "C456",
            "oldbalanceDest": "0",
            "newbalanceDest": "1000"
        }

        response = self.client.post('/predict', data=test_data)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b'Fraud Transaction' in response.data or b'Legitimate' in response.data
        )

if __name__ == "__main__":
    unittest.main()
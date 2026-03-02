import unittest
from flask_app.app import app

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Fraud Detection System</title>', response.data)

    def test_predict_page(self):
        response = self.client.post('/predict', data={
            "type": "TRANSFER",
            "amount": 1000,
            "oldbalanceOrg": 5000,
            "newbalanceOrig": 4000,
            "oldbalanceDest": 0,
            "newbalanceDest": 1000
        })
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b'Fraud' in response.data or b'Legitimate' in response.data
        )

if __name__ == '__main__':
    unittest.main()
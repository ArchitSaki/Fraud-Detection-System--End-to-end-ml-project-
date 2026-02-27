import pandas as pd
import boto3
from src.logger import logging
from io import StringIO


class S3_operations:
    def __init__(self, bucket_name, aws_access_key, aws_secret_key, region="us-east-1"):
        self.bucket_name = bucket_name

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )

        logging.info("Successfully connected to AWS S3")

    def get_file_s3(self, file_name):
        try:
            logging.info(f"Fetching file {file_name} from bucket {self.bucket_name}")

            obj = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=file_name
            )

            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

            logging.info(f"Successfully loaded dataframe from S3 bucket {self.bucket_name}")
            return df

        except Exception as e:
            logging.error(f"Error fetching {file_name} from {self.bucket_name}: {e}")
            raise
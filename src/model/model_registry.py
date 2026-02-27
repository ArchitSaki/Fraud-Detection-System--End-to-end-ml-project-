import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

mlflow.set_tracking_uri('https://dagshub.com/ArchitSaki/Fraud-Detection-System--End-to-end-ml-project-.mlflow')
dagshub.init(repo_owner='ArchitSaki', repo_name='Fraud-Detection-System--End-to-end-ml-project-', mlflow=True)

def load_model_info(file_path):
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register(model_name,model_info):
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        model_version = mlflow.register_model(model_uri, model_name)
        
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def model_register (run_id : str , model_path : str , model_name : str) ->None:
    try:
        # model_uri = f"runs:/{run_id}/{model_path}"
        # logging.info("registering model")
        # model_version = mlflow.register_model(model_uri, model_name)
        # logging.info("model registered")

        # logging.info("moving model to staging")
        
        client = mlflow.tracking.MlflowClient()

        run = client.get_run(run_id)
        model_uri = f"{run.info.artifact_uri}/{model_path}"
        logging.info("registering model")
        registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
        logging.info(f"model registered version {registered_model.version}")
        client.transition_model_version_stage(
            name=model_name,
            version=registered_model.version,
            stage="Staging"
        )
        logging.info("model moved to stagging")
        
    except Exception as e:
        logging.error(f"error {e} while registering model ...")
        raise e

# def main():
#     try:
#         model_info_path = 'reports/experiment_info.json'
#         model_info = load_model_info(model_info_path)
        
#         model_name = "my_model"
#         register(model_name, model_info)
#     except Exception as e:
#         logging.error('Failed to complete the model registration process: %s', e)
#         print(f"Error: {e}")

# if __name__ == '__main__':
#     main()
def main () :
    try:
        model_info = load_model_info (file_path="./reports/experiment_info.json")

        run_id = model_info["run_id"]
        model_path =  model_info["model_path"]
        model_name = "my_model"
        model_register(run_id=run_id , model_path=model_path , model_name=model_name)
    except Exception as e:
        logging.error(f"error {e} in model registeration...")
        raise e
    
if __name__ == '__main__':
    main()
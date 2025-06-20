# egctools/kaggle.py

# This module contains functions for working with Kaggle datasets.

# Package & Libraries
import os
import zipfile
import json
import getpass
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

def create_kaggle_token_dict():
    """
    Prompt the user to enter their Kaggle username and key, and return a dictionary.
    
    Returns:
    dict: A dictionary containing the Kaggle username and key.
    """
    print("To create a Kaggle API token dictionary, follow these steps:\n")
    print("1. Go to Kaggle's website and log in: https://www.kaggle.com/")
    print("2. Click on your profile picture in the top right corner and select 'My Account'.")
    print("3. Scroll down to the 'API' section and click on 'Create New API Token'. This will download a file named 'kaggle.json'.")
    print("4. Open the 'kaggle.json' file with a text editor. It contains your username and key.\n")
    print("Now, enter your Kaggle credentials:")
    
    username = input("Enter your Kaggle username: ")
    key = getpass.getpass("Enter your Kaggle API key: ")
    
    kaggle_token = {
        "username": username,
        "key": key
    }
    
    print("\nYour Kaggle API token dictionary has been created.")
    return kaggle_token

def load_kaggle_dataset(dataset_name: str, data_dir: str = 'data', kaggle_token: dict = None):
    """
    Download and extract a Kaggle dataset.

    Parameters:
    dataset_name (str): The name of the Kaggle dataset (e.g., 'yasserh/housing-prices-dataset').
    data_dir (str): The directory where the extracted files will be stored. Default is 'data'.
    kaggle_token (dict): Optional Kaggle API token. If provided, it should be a dictionary with keys 'username' and 'key'.

    Returns:
    str: The file path of the extracted dataset CSV file.
    """
    # Setup Kaggle API token if provided
    if kaggle_token:
        kaggle_config_dir = os.path.expanduser('~/.kaggle')
        os.makedirs(kaggle_config_dir, exist_ok=True)
        with open(os.path.join(kaggle_config_dir, 'kaggle.json'), 'w') as f:
            json.dump(kaggle_token, f)
        os.chmod(os.path.join(kaggle_config_dir, 'kaggle.json'), 0o600)

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Download the dataset
    try:
        kaggle.api.dataset_download_files(dataset_name, path='.', unzip=False)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None
    
    # Find the downloaded zip file
    zipfile_name = next(f for f in os.listdir() if f.endswith('.zip'))
    print(f"Extracting {zipfile_name}")
    
    # Extract the zip file
    with zipfile.ZipFile(zipfile_name, 'r') as zip_ref:
        zip_ref.extractall()

    # Move CSV files to the data directory
    for file in os.listdir():
        if file.endswith('.csv'):
            os.rename(file, os.path.join(data_dir, file))
    
    # Cleanup
    os.remove(zipfile_name)
    print("Successfully downloaded and extracted dataset")
    
    # Automatically detect dataset filename, save filepath of dataset
    dataset_filename = next(f for f in os.listdir(data_dir) if f.endswith('.csv'))
    dataset_filepath = os.path.join(data_dir, dataset_filename)
    print(f"Dataset file path: {dataset_filepath}")

    return dataset_filepath


import os
import pandas as pd
import requests
import zipfile
import io
import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# --- Dataset URLs ---
# Using a reliable mirror for Northwind
NORTHWIND_URL = "http://bitnine.net/tutorial/import-northwind-dataset.zip"
HR_DATASET_URL = "https://www.kaggle.com/datasets/rhuebner/human-resources-data-set/download?datasetVersionNumber=2"
FINANCIAL_TRANSACTIONS_URL = "https://www.kaggle.com/datasets/garfield18/credit-card-transactions/download?datasetVersionNumber=1"


def download_and_extract_zip(url: str, dest_path: str):
    """Downloads and extracts a zip file from a URL."""
    if os.path.exists(dest_path):
        logger.info(f"Dataset already exists at {dest_path}. Skipping download.")
        return
    
    logger.info(f"Downloading dataset from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(dest_path)
        logger.info(f"Successfully downloaded and extracted to {dest_path}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download dataset: {e}")
        raise
    except zipfile.BadZipFile:
        logger.error("Failed to extract zip file. The downloaded file may be corrupt or not a zip file.")
        # Handle Kaggle's HTML response if not logged in
        logger.error("If downloading from Kaggle, make sure you are logged in or provide kaggle.json.")
        raise


def process_northwind_dataset(data_path: str) -> List[Tuple[str, str, Any]]:
    """Processes the Northwind dataset into (field_name, type, value) triples."""
    triples = []
    
    all_csv_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.csv'):
                all_csv_files.append(os.path.join(root, file))

    if not all_csv_files:
        logger.warning(f"No CSV files found in {data_path} for Northwind dataset.")
        return []
        
    for file_path in all_csv_files:
        try:
            df = pd.read_csv(file_path)
            for col in df.columns:
                for value in df[col].dropna():
                    field_name = col.lower()
                    value_type = str(df[col].dtype)
                    triples.append((field_name, value_type, value))
        except Exception as e:
            logger.warning(f"Could not process {file_path}: {e}")
            
    return triples


def process_hr_dataset(data_path: str) -> List[Tuple[str, str, Any]]:
    """Processes the HR dataset into (field_name, type, value) triples."""
    # This function will need to be adapted based on the actual file name
    # after downloading from Kaggle. Assuming a file named 'HRDataset_v14.csv'.
    file_path = os.path.join(data_path, "HRDataset_v14.csv")
    if not os.path.exists(file_path):
        logger.warning(f"HR dataset file not found at {file_path}. Please check the filename.")
        return []
        
    triples = []
    try:
        df = pd.read_csv(file_path)
        for col in df.columns:
            for value in df[col].dropna():
                field_name = col.lower().replace(" ", "_")
                value_type = str(df[col].dtype)
                triples.append((field_name, value_type, value))
    except Exception as e:
        logger.error(f"Failed to process HR dataset: {e}")
        
    return triples


def process_financial_dataset(data_path: str) -> List[Tuple[str, str, Any]]:
    """Processes the Financial Transactions dataset into triples."""
    # This will also need adaptation based on the actual filename.
    # Assuming 'credit_card_transactions.csv'.
    file_path = os.path.join(data_path, "credit_card_transactions.csv")
    if not os.path.exists(file_path):
        logger.warning(f"Financial dataset file not found at {file_path}. Please check the filename.")
        return []
        
    triples = []
    try:
        df = pd.read_csv(file_path)
        for col in df.columns:
            for value in df[col].dropna():
                field_name = col.lower()
                value_type = str(df[col].dtype)
                triples.append((field_name, value_type, value))
    except Exception as e:
        logger.error(f"Failed to process financial dataset: {e}")
        
    return triples


def create_training_data(output_path: str, data_dir: str = "data"):
    """
    Main function to download, process, and save the training data.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # --- Download Datasets ---
    northwind_path = os.path.join(data_dir, "northwind")
    hr_path = os.path.join(data_dir, "hr_dataset")
    financial_path = os.path.join(data_dir, "financial_dataset")
    
    try:
        download_and_extract_zip(NORTHWIND_URL, northwind_path)
    except Exception as e:
        logger.warning(f"Failed to automatically download Northwind dataset: {e}")
        logger.warning(f"Please manually download a Northwind CSV dataset (e.g., from GitHub) and extract its contents into the '{northwind_path}' directory.")
    
    # Kaggle downloads require authentication, so we'll need to instruct the user.
    logger.info("To download Kaggle datasets, please ensure your kaggle.json is in ~/.kaggle/")
    logger.info(f"Download HR dataset from: {HR_DATASET_URL}")
    logger.info(f"Download Financial dataset from: {FINANCIAL_TRANSACTIONS_URL}")
    logger.info(f"And place them in {hr_path} and {financial_path} respectively.")

    # --- Process Datasets ---
    all_triples = []
    
    if os.path.exists(northwind_path):
        logger.info("Processing Northwind dataset...")
        all_triples.extend(process_northwind_dataset(northwind_path))
        
    if os.path.exists(hr_path):
        logger.info("Processing HR dataset...")
        all_triples.extend(process_hr_dataset(hr_path))
        
    if os.path.exists(financial_path):
        logger.info("Processing Financial dataset...")
        all_triples.extend(process_financial_dataset(financial_path))
        
    if not all_triples:
        logger.warning("No data was processed. Training data creation failed.")
        return

    # --- Save to DataFrame ---
    df = pd.DataFrame(all_triples, columns=["field_name", "type", "value"])
    
    # Basic cleaning
    df.drop_duplicates(inplace=True)
    df = df[df['value'].astype(str).str.len() > 0]
    
    # Save to a single file
    df.to_csv(output_path, index=False)
    logger.info(f"Training data successfully created at {output_path}")


if __name__ == "__main__":
    # Example of how to run this script
    logging.basicConfig(level=logging.INFO)
    create_training_data("training_data.csv")

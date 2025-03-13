import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# File handler
log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f'Parameters retrieved from {params_path}')
        return params
    except FileNotFoundError:
        logger.error(f'File not found: {params_path}. Creating a default params.yaml file.')
        default_params = {
            'data_ingestion': {'test_size': 0.20},
            'feature_engineering': {'max_features': 35},
            'model_building': {'n_estimators': 22, 'random_state': 2}
        }
        with open(params_path, 'w') as file:
            yaml.safe_dump(default_params, file)
        logger.info(f'Default params.yaml created at {params_path}')
        return default_params
    except yaml.YAMLError as e:
        logger.error(f'YAML error: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill missing values with empty strings
        logger.debug(f'Data loaded and NaNs filled from {file_path}')
        return df
    except FileNotFoundError:
        logger.error(f'File not found: {file_path}')
        raise
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse the CSV file: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error occurred while loading the data: {e}')
        raise

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """Apply TF-IDF to the data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        # Extract text and target columns
        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        # Transform the text data
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        # Convert to DataFrames
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        logger.debug('TF-IDF applied and data transformed')
        return train_df, test_df
    except Exception as e:
        logger.error(f'Error during TF-IDF transformation: {e}')
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the DataFrame to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug(f'Data saved to {file_path}')
    except Exception as e:
        logger.error(f'Unexpected error occurred while saving the data: {e}')
        raise

def main():
    try:
        # Load parameters
        params = load_params(params_path='params.yaml')
        max_features = params['feature_engineering']['max_features']

        # Load data
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        # Apply TF-IDF
        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        # Save processed data
        save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))

        logger.info('Feature engineering process completed successfully.')
    except Exception as e:
        logger.error(f'Failed to complete the feature engineering process: {e}')
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

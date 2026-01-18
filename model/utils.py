"""
Utility functions for data preprocessing and validation
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def validate_dataset(df):
    """
    Validate dataset meets minimum requirements

    Args:
        df: pandas DataFrame

    Returns:
        tuple: (is_valid, message)
    """
    n_rows, n_cols = df.shape

    # Check minimum instances
    if n_rows < 500:
        return False, f"Dataset has only {n_rows} instances. Minimum required: 500"

    # Check minimum features (excluding target)
    if n_cols < 12:
        return False, f"Dataset has only {n_cols} columns. Minimum required: 12 (including target)"

    return True, f"Dataset validated: {n_rows} instances, {n_cols} features"


def preprocess_data(df, target_column, test_size=0.2, random_state=42):
    """
    Preprocess data for ML models

    Args:
        df: pandas DataFrame
        target_column: name of target column
        test_size: proportion of test set
        random_state: random seed

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, label_encoder)
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Store original feature names
    feature_names = X.columns.tolist()

    # Encode categorical features
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Encode target if categorical
    label_encoder = None
    if y.dtype == 'object' or not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # Convert to numeric (handle any remaining issues)
    X = X.apply(pd.to_numeric, errors='coerce')

    # Fill NaN values with median
    X = X.fillna(X.median())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_names, label_encoder


def get_sample_dataset():
    """
    Load Spambase dataset from UCI Machine Learning Repository

    Dataset Details:
    - Source: UCI Machine Learning Repository
    - URL: https://archive.ics.uci.edu/ml/datasets/spambase
    - Instances: 4,601 email messages
    - Features: 57 (word frequencies, character frequencies, capital letter metrics)
    - Classes: 2 (spam=1, not spam=0)
    - Type: Binary classification
    - Application: Email spam detection

    Features include:
    - 48 word frequency features (frequency of specific words)
    - 6 character frequency features (frequency of special characters)
    - 3 capital letter features (run length statistics)

    Citation:
    Mark Hopkins, Erik Reeber, George Forman, Jaap Suermondt
    Hewlett-Packard Labs, 1501 Page Mill Rd., Palo Alto, CA 94304
    Donated to UCI Repository in 1999

    Returns:
        pandas DataFrame with spambase data
    """
    import warnings
    warnings.filterwarnings('ignore')

    # Try loading from UCI repository
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"

        # Define column names (57 features + 1 target)
        # Word frequency features (0-47)
        word_features = [
            'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
            'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
            'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
            'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
            'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
            'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money',
            'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650',
            'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857',
            'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology',
            'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
            'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
            'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference'
        ]

        # Character frequency features (48-53)
        char_features = [
            'char_freq_semicolon', 'char_freq_parenthesis', 'char_freq_bracket',
            'char_freq_exclamation', 'char_freq_dollar', 'char_freq_hash'
        ]

        # Capital letter features (54-56)
        capital_features = [
            'capital_run_length_average',
            'capital_run_length_longest',
            'capital_run_length_total'
        ]

        # All column names
        column_names = word_features + char_features + capital_features + ['target']

        # Load dataset
        df = pd.read_csv(url, header=None, names=column_names)

        # Ensure target is integer
        df['target'] = df['target'].astype(int)

        return df

    except Exception as e:
        print(f"⚠️  Could not load from UCI repository: {e}")
        print(f"⚠️  Using fallback Wine dataset...")

        # Fallback: Use sklearn's built-in wine dataset
        from sklearn.datasets import load_wine

        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target

        return df


"""
Utility functions for data preprocessing and validation
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def get_expected_spambase_columns():
    """
    Get the expected column names for spambase dataset format

    Returns:
        list: Expected column names (57 features + 1 target = 58 columns)
    """
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

    char_features = [
        'char_freq_semicolon', 'char_freq_parenthesis', 'char_freq_bracket',
        'char_freq_exclamation', 'char_freq_dollar', 'char_freq_hash'
    ]

    capital_features = [
        'capital_run_length_average',
        'capital_run_length_longest',
        'capital_run_length_total'
    ]

    return word_features + char_features + capital_features + ['target']


def validate_dataset(df, strict=True):
    """
    Validate dataset - STRICT MODE: Only allow spambase format

    Args:
        df: pandas DataFrame
        strict: If True, only allow exact spambase format (default: True)

    Returns:
        tuple: (is_valid, message, warnings_list)
    """
    n_rows, n_cols = df.shape
    warnings = []

    # Get expected columns
    expected_columns = get_expected_spambase_columns()

    # STRICT VALIDATION: Only allow spambase format
    if strict:
        # Check exact number of columns
        if n_cols != len(expected_columns):
            return False, f"Dataset must have exactly {len(expected_columns)} columns (spambase format). Found: {n_cols} columns", []

        # Check exact column names
        if list(df.columns) != expected_columns:
            missing_cols = set(expected_columns) - set(df.columns)
            extra_cols = set(df.columns) - set(expected_columns)

            error_msg = "Dataset must match spambase format exactly."
            if missing_cols:
                error_msg += f" Missing columns: {', '.join(list(missing_cols)[:5])}"
            if extra_cols:
                error_msg += f" Extra columns: {', '.join(list(extra_cols)[:5])}"

            return False, error_msg, []

        # Check if target column has at least 2 classes
        if 'target' in df.columns:
            target_counts = df['target'].value_counts()
            if len(target_counts) < 2:
                return False, f"Target column must have at least 2 classes. Found: {len(target_counts)}", []

            # Check if each class has at least 2 samples (needed for stratified split)
            min_class_count = target_counts.min()
            if min_class_count < 2:
                return False, f"Each target class must have at least 2 samples. Smallest class has only {min_class_count} sample(s)", []

    # Check minimum instances
    if n_rows < 100:
        return False, f"Dataset must have at least 100 instances. Found: {n_rows}", []

    return True, f"Dataset validated: {n_rows} instances, {n_cols} features", []


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
    # Validate target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset. Available columns: {df.columns.tolist()}")

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

    # Check if stratification is possible
    target_counts = pd.Series(y).value_counts()
    can_stratify = len(target_counts) >= 2 and target_counts.min() >= 2

    if not can_stratify:
        raise ValueError(
            f"Cannot perform stratified split. Each class must have at least 2 samples. "
            f"Current class distribution: {target_counts.to_dict()}"
        )

    # Split data with stratification
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
    Load Spambase dataset from local file (for faster loading)

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
    import os
    warnings.filterwarnings('ignore')

    # Try loading from local file first (much faster!)
    local_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'spambase.csv')

    try:
        if os.path.exists(local_path):
            # Load from local CSV (fast!)
            df = pd.read_csv(local_path)
            df['target'] = df['target'].astype(int)
            return df
    except Exception as e:
        print(f"⚠️  Could not load from local file: {e}")

    # Fallback: Try loading from UCI repository
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"

        # Define column names (57 features + 1 target)
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

        char_features = [
            'char_freq_semicolon', 'char_freq_parenthesis', 'char_freq_bracket',
            'char_freq_exclamation', 'char_freq_dollar', 'char_freq_hash'
        ]

        capital_features = [
            'capital_run_length_average',
            'capital_run_length_longest',
            'capital_run_length_total'
        ]

        column_names = word_features + char_features + capital_features + ['target']

        # Load dataset from UCI
        df = pd.read_csv(url, header=None, names=column_names)
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


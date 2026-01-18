"""
ML Classification Models Implementation
"""
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from .metrics import calculate_all_metrics, get_confusion_matrix


class ModelTrainer:
    """
    Class to train and evaluate multiple classification models
    """

    def __init__(self, random_state=42):
        """
        Initialize all models

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                solver='lbfgs'
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=random_state,
                max_depth=10,
                min_samples_split=5
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                metric='euclidean'
            ),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(
                random_state=random_state,
                n_estimators=100,
                max_depth=15,
                min_samples_split=5
            ),
            'XGBoost': XGBClassifier(
                random_state=random_state,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                eval_metric='logloss'
            )
        }
        self.trained_models = {}
        self.results = {}

    def train_model(self, model_name, X_train, y_train):
        """
        Train a specific model

        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model

        return model

    def save_model(self, model_name, filepath=None):
        """
        Save a trained model to a pickle file

        Args:
            model_name: Name of the model to save
            filepath: Path to save the model (optional, auto-generated if None)

        Returns:
            str: Path where model was saved
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")

        # Create model directory if it doesn't exist
        model_dir = 'model/saved_models'
        os.makedirs(model_dir, exist_ok=True)

        # Generate filename if not provided
        if filepath is None:
            # Clean model name for filename
            clean_name = model_name.lower().replace(' ', '_')
            filepath = os.path.join(model_dir, f'{clean_name}.pkl')

        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(self.trained_models[model_name], f)

        return filepath

    def load_model(self, model_name, filepath):
        """
        Load a trained model from a pickle file

        Args:
            model_name: Name to assign to the loaded model
            filepath: Path to the pickle file

        Returns:
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)

        self.trained_models[model_name] = model
        return model

    def save_all_models(self, directory='model/saved_models'):
        """
        Save all trained models to pickle files

        Args:
            directory: Directory to save models (default: model/saved_models)

        Returns:
            dict: Dictionary with model names and their save paths
        """
        os.makedirs(directory, exist_ok=True)
        saved_paths = {}

        for model_name in self.trained_models.keys():
            clean_name = model_name.lower().replace(' ', '_')
            filepath = os.path.join(directory, f'{clean_name}.pkl')

            with open(filepath, 'wb') as f:
                pickle.dump(self.trained_models[model_name], f)

            saved_paths[model_name] = filepath

        return saved_paths

    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate a trained model

        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test labels

        Returns:
            dict: Evaluation metrics and predictions
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")

        model = self.trained_models[model_name]

        # Predictions
        y_pred = model.predict(X_test)

        # Predicted probabilities
        try:
            y_pred_proba = model.predict_proba(X_test)
        except:
            y_pred_proba = None

        # Calculate metrics
        metrics = calculate_all_metrics(y_test, y_pred, y_pred_proba)

        # Confusion matrix
        cm = get_confusion_matrix(y_test, y_pred)

        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm
        }

        return self.results[model_name]

    def train_and_evaluate_all(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate all models

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels

        Returns:
            dict: Results for all models
        """
        for model_name in self.models.keys():
            print(f"Training {model_name}...")
            self.train_model(model_name, X_train, y_train)
            self.evaluate_model(model_name, X_test, y_test)

        return self.results

    def get_results_dataframe(self):
        """
        Get results as a pandas DataFrame

        Returns:
            pandas DataFrame: Results table
        """
        if not self.results:
            return pd.DataFrame()

        results_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            row = {
                'ML Model Name': model_name,
                'Accuracy': f"{metrics['Accuracy']:.4f}",
                'AUC': f"{metrics['AUC']:.4f}",
                'Precision': f"{metrics['Precision']:.4f}",
                'Recall': f"{metrics['Recall']:.4f}",
                'F1': f"{metrics['F1']:.4f}",
                'MCC': f"{metrics['MCC']:.4f}"
            }
            results_data.append(row)

        return pd.DataFrame(results_data)

    def get_model_observations(self):
        """
        Generate observations about model performance

        Returns:
            dict: Model observations
        """
        if not self.results:
            return {}

        observations = {
            'Logistic Regression': "Linear model performing well for linearly separable data. Fast training and interpretable coefficients. May underfit complex non-linear patterns.",
            'Decision Tree': "Captures non-linear relationships effectively. Prone to overfitting without proper pruning. Highly interpretable with feature importance.",
            'KNN': "Non-parametric lazy learner. Performance depends on distance metric and K value. Can be slow on large datasets. Sensitive to feature scaling.",
            'Naive Bayes': "Probabilistic classifier assuming feature independence. Fast and efficient. Works well with small datasets. May underperform if independence assumption violated.",
            'Random Forest': "Ensemble method reducing overfitting through bagging. Robust and generally high accuracy. Provides feature importance. Less interpretable than single trees.",
            'XGBoost': "Gradient boosting ensemble with regularization. Often achieves highest accuracy. Handles missing values well. Requires careful hyperparameter tuning."
        }

        # Add performance-specific observations
        results_df = self.get_results_dataframe()
        if not results_df.empty:
            results_df['Accuracy_num'] = results_df['Accuracy'].astype(float)
            best_model = results_df.loc[results_df['Accuracy_num'].idxmax(), 'ML Model Name']

            for model_name in observations.keys():
                perf = "High" if model_name == best_model else "Moderate"
                observations[model_name] = f"{observations[model_name]} Performance on this dataset: {perf}."

        return observations


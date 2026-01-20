"""
Machine Learning Classification Models - Training & Evaluation

This script demonstrates the implementation and evaluation of 6 classification models:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Random Forest
6. XGBoost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from classifier import ModelTrainer
from utils import validate_dataset, preprocess_data, get_sample_dataset
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def main():
    """Main function to train and evaluate all models"""

    print("=" * 80)
    print("MACHINE LEARNING CLASSIFICATION MODELS - TRAINING SCRIPT")
    print("=" * 80 + "\n")

    # ========================================================================
    # Step 1: Load and Explore Dataset
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATASET")
    print("=" * 80 + "\n")

    # Load sample dataset (or load your own CSV)
    # df = pd.read_csv('your_dataset.csv')
    df = get_sample_dataset()

    print(f"Dataset Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())

    # Dataset information
    print("\nDataset Info:")
    df.info()

    # Statistical summary
    print("\nStatistical Summary:")
    print(df.describe())

    # Validate dataset
    is_valid, message, warnings_list = validate_dataset(df, strict=True)
    print(f"\nValidation: {message}")
    if warnings_list:
        print("⚠️  Warnings:")
        for warning in warnings_list:
            print(f"   - {warning}")

    # Target distribution
    target_col = 'target'
    print("\nTarget Variable Distribution:")
    print(df[target_col].value_counts())

    plt.figure(figsize=(8, 5))
    df[target_col].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Target Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: target_distribution.png")

    # ========================================================================
    # Step 2: Data Preprocessing
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 80 + "\n")

    X_train, X_test, y_train, y_test, feature_names, label_encoder = preprocess_data(
        df, target_col, test_size=0.2, random_state=42
    )

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Feature names: {feature_names}")

    # ========================================================================
    # Step 3: Train All Models
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING ALL MODELS")
    print("=" * 80 + "\n")

    trainer = ModelTrainer(random_state=42)

    print("Training all models...\n")
    results = trainer.train_and_evaluate_all(X_train, X_test, y_train, y_test)

    print("\n✅ All models trained successfully!")

    # ========================================================================
    # Step 4: Evaluation Metrics Comparison
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: EVALUATION METRICS COMPARISON")
    print("=" * 80 + "\n")

    results_df = trainer.get_results_dataframe()
    print("Model Performance Comparison:")
    print(results_df.to_string(index=False))

    # Save results to CSV
    results_df.to_csv('model_results.csv', index=False)
    print("\n✅ Results saved to 'model_results.csv'")

    # ========================================================================
    # Step 5: Visualize Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: VISUALIZING RESULTS")
    print("=" * 80 + "\n")

    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    models = results_df['ML Model Name'].tolist()

    # Convert to numeric
    plot_data = results_df.copy()
    for metric in metrics:
        plot_data[metric] = plot_data[metric].astype(float)

    # Grouped bar chart
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Performance Across Different Metrics', fontsize=16, fontweight='bold')

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        plot_data.plot(x='ML Model Name', y=metric, kind='bar', ax=ax, legend=False, color='steelblue')
        ax.set_title(f'{metric} Comparison', fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: metrics_comparison.png")

    # Heatmap of all metrics
    plt.figure(figsize=(10, 6))
    heatmap_data = plot_data.set_index('ML Model Name')[metrics].astype(float)
    sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='YlGnBu', cbar_kws={'label': 'Score'})
    plt.title('Model Performance Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Model')
    plt.ylabel('Metric')
    plt.tight_layout()
    plt.savefig('performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: performance_heatmap.png")

    # ========================================================================
    # Step 6: Confusion Matrices
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: GENERATING CONFUSION MATRICES")
    print("=" * 80 + "\n")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Confusion Matrices for All Models', fontsize=14, fontweight='bold')

    for idx, model_name in enumerate(models):
        ax = axes[idx // 3, idx % 3]
        cm = trainer.results[model_name]['confusion_matrix']

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    cbar_kws={'label': 'Count'}, annot_kws={'size': 10})
        ax.set_title(model_name, fontweight='bold', fontsize=11)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: confusion_matrices.png")

    # ========================================================================
    # Step 7: Model Observations
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: MODEL OBSERVATIONS")
    print("=" * 80 + "\n")

    observations = trainer.get_model_observations()

    for model_name, observation in observations.items():
        print(f"📊 {model_name}:")
        print(f"   {observation}\n")

    # Create observations DataFrame
    obs_df = pd.DataFrame([
        {'ML Model Name': model, 'Observation about model performance': obs}
        for model, obs in observations.items()
    ])

    print("\nObservations Table:")
    print(obs_df.to_string(index=False))

    # Save observations
    obs_df.to_csv('model_observations.csv', index=False)
    print("\n✅ Observations saved to 'model_observations.csv'")

    # ========================================================================
    # Step 8: Best Model Identification
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 8: BEST MODELS PER METRIC")
    print("=" * 80 + "\n")

    for metric in metrics:
        best_idx = plot_data[metric].astype(float).idxmax()
        best_model = plot_data.loc[best_idx, 'ML Model Name']
        best_score = plot_data.loc[best_idx, metric]
        print(f"🏆 Best {metric:12s}: {best_model:20s} (Score: {best_score})")

    # ========================================================================
    # Conclusion
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("\nThis script demonstrated the complete workflow for:")
    print("1. ✅ Loading and validating classification datasets")
    print("2. ✅ Preprocessing data (encoding, scaling, splitting)")
    print("3. ✅ Training 6 different classification models")
    print("4. ✅ Evaluating models with 6 comprehensive metrics")
    print("5. ✅ Visualizing and comparing model performance")
    print("6. ✅ Generating insights and observations")
    print("\nAll results have been saved to CSV files and PNG images.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()


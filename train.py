"""
Standalone script to train and evaluate all models
Run this script to test the implementation without Streamlit
"""

import pandas as pd
import numpy as np
from model.classifier import ModelTrainer
from model.utils import validate_dataset, preprocess_data, get_sample_dataset
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 80)
    print("ML CLASSIFICATION MODELS - TRAINING & EVALUATION")
    print("=" * 80)
    print()

    # Load dataset
    print("📊 Loading dataset...")
    df = get_sample_dataset()
    print(f"✅ Dataset loaded: {df.shape[0]} instances, {df.shape[1]} features")
    print()

    # Validate dataset
    _, message = validate_dataset(df)
    print(f"🔍 Validation: {message}")
    print()

    # Display dataset info
    print("Dataset Information:")
    print("-" * 80)
    print(df.info())
    print()

    print("Target Distribution:")
    print("-" * 80)
    print(df['target'].value_counts())
    print()

    # Preprocess data
    print("🔧 Preprocessing data...")
    target_col = 'target'
    X_train, X_test, y_train, y_test, feature_names, _ = preprocess_data(
        df, target_col, test_size=0.2, random_state=42
    )
    print(f"✅ Training set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")
    print(f"✅ Features: {len(feature_names)}")
    print()

    # Initialize trainer
    print("🤖 Initializing models...")
    trainer = ModelTrainer(random_state=42)
    print(f"✅ Initialized {len(trainer.models)} models")
    print()

    # Train and evaluate all models
    print("🚀 Training and evaluating all models...")
    print("-" * 80)

    for idx, model_name in enumerate(trainer.models.keys(), 1):
        print(f"\n[{idx}/6] Training {model_name}...")
        trainer.train_model(model_name, X_train, y_train)
        trainer.evaluate_model(model_name, X_test, y_test)
        print(f"✅ {model_name} completed")

    print()
    print("=" * 80)
    print("RESULTS - MODEL PERFORMANCE COMPARISON")
    print("=" * 80)
    print()

    # Get results DataFrame
    results_df = trainer.get_results_dataframe()

    # Display results
    print(results_df.to_string(index=False))
    print()

    # Save results
    results_df.to_csv('model_results.csv', index=False)
    print("💾 Results saved to 'model_results.csv'")
    print()

    # Get observations
    print("=" * 80)
    print("MODEL OBSERVATIONS")
    print("=" * 80)
    print()

    observations = trainer.get_model_observations()
    for model_name, observation in observations.items():
        print(f"📊 {model_name}:")
        print(f"   {observation}")
        print()

    # Save observations
    obs_df = pd.DataFrame([
        {'ML Model Name': model, 'Observation about model performance': obs}
        for model, obs in observations.items()
    ])
    obs_df.to_csv('model_observations.csv', index=False)
    print("💾 Observations saved to 'model_observations.csv'")
    print()

    # Save trained models
    print("=" * 80)
    print("SAVING TRAINED MODELS")
    print("=" * 80)
    print()

    saved_paths = trainer.save_all_models()
    for model_name, path in saved_paths.items():
        print(f"💾 {model_name:25s} → {path}")
    print()
    print(f"✅ All {len(saved_paths)} models saved successfully!")
    print()

    # Find best models
    print("=" * 80)
    print("BEST MODELS PER METRIC")
    print("=" * 80)
    print()

    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    for metric in metrics:
        results_df[f'{metric}_num'] = results_df[metric].astype(float)
        best_idx = results_df[f'{metric}_num'].idxmax()
        best_model = results_df.loc[best_idx, 'ML Model Name']
        best_score = results_df.loc[best_idx, metric]
        print(f"🏆 Best {metric:12s}: {best_model:25s} (Score: {best_score})")

    print()
    print("=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print("Generated Files:")
    print("  - model_results.csv (comparison table)")
    print("  - model_observations.csv (observations table)")
    print("  - model/saved_models/*.pkl (6 trained models)")
    print()
    print("Next Steps:")
    print("  1. Review the results above")
    print("  2. Run 'streamlit run app.py' for interactive dashboard")
    print("  3. Upload to GitHub")
    print("  4. Deploy on Streamlit Cloud")
    print()

if __name__ == "__main__":
    main()


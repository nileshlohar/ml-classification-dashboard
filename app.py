"""
Streamlit Web Application for ML Classification Models
Interactive ML Model Comparison Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from model.classifier import ModelTrainer
from model.utils import validate_dataset, preprocess_data, get_sample_dataset
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        padding: 0.5rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🤖 ML Classification Model Comparison Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Configuration Panel")

    # Data source selection
    st.markdown("#### 1️⃣ Data Source")
    data_source = st.radio(
        "Choose data source:",
        ["Use Sample Dataset", "Upload CSV File"],
        index=0,  # Default to "Use Sample Dataset"
        help="Upload your own dataset or use the built-in sample dataset"
    )

    # File upload
    uploaded_file = None
    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "Upload your dataset (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="Minimum 500 instances and 12 features required"
        )

    st.markdown("---")

    # Model selection
    st.markdown("#### 2️⃣ Model Selection")
    model_options = [
        'Logistic Regression',
        'Decision Tree',
        'KNN',
        'Naive Bayes',
        'Random Forest',
        'XGBoost'
    ]

    selected_models = st.multiselect(
        "Select models to train:",
        model_options,
        default=model_options,
        help="Select one or more models for comparison"
    )

    st.markdown("---")

    # Training parameters
    st.markdown("#### 3️⃣ Training Parameters")
    test_size = st.slider(
        "Test Set Size (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        help="Percentage of data to use for testing"
    )

    random_seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=1000,
        value=42,
        help="Set random seed for reproducibility"
    )

    st.markdown("---")

    # About section
    with st.expander("ℹ️ About"):
        st.markdown("""
        **ML Models Implemented:**
        1. Logistic Regression
        2. Decision Tree Classifier
        3. K-Nearest Neighbors (KNN)
        4. Naive Bayes (Gaussian)
        5. Random Forest (Ensemble)
        6. XGBoost (Ensemble)

        **Evaluation Metrics:**
        - Accuracy
        - AUC Score
        - Precision
        - Recall
        - F1 Score
        - MCC (Matthews Correlation Coefficient)
        """)

# Main content area
def load_data(uploaded_file, use_sample):
    """Load and validate dataset"""
    try:
        if use_sample:
            df = get_sample_dataset()
            st.success("✅ Sample dataset loaded successfully!")
        else:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success(f"✅ Dataset '{uploaded_file.name}' loaded successfully!")

        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None

# Load dataset
df = None
if data_source == "Use Sample Dataset":
    # Auto-load sample dataset if not already loaded or if user switched back to it
    if not st.session_state.data_loaded or 'is_sample' not in st.session_state or not st.session_state.is_sample:
        df = load_data(None, True)
        if df is not None:
            st.session_state.data_loaded = True
            st.session_state.df = df
            st.session_state.is_sample = True
elif uploaded_file is not None:
    # Load uploaded file
    df = load_data(uploaded_file, False)
    if df is not None:
        st.session_state.data_loaded = True
        st.session_state.df = df
        st.session_state.is_sample = False
else:
    # User selected upload but hasn't uploaded a file yet
    st.session_state.data_loaded = False
    st.session_state.is_sample = False

# Display dataset info
if st.session_state.data_loaded:
    df = st.session_state.df

    st.markdown('<h2 class="sub-header">📋 Dataset Overview</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Instances", df.shape[0])
    with col2:
        st.metric("Total Features", df.shape[1])
    with col3:
        st.metric("Numeric Features", df.select_dtypes(include=[np.number]).shape[1])
    with col4:
        st.metric("Categorical Features", df.select_dtypes(include=['object']).shape[1])

    # Validate dataset
    is_valid, message = validate_dataset(df)
    if is_valid:
        st.success(f"✅ {message}")
    else:
        st.warning(f"⚠️ {message}")

    # Display dataset
    with st.expander("📊 View Dataset", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)

    # Dataset statistics
    with st.expander("📈 Dataset Statistics"):
        st.write(df.describe())

    st.markdown("---")

    # Target column selection
    st.markdown('<h2 class="sub-header">🎯 Target Variable Selection</h2>', unsafe_allow_html=True)
    target_column = st.selectbox(
        "Select the target column:",
        df.columns.tolist(),
        index=len(df.columns) - 1,
        help="Choose the column you want to predict"
    )

    # Display target distribution
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Target Variable Distribution:**")
        target_counts = df[target_column].value_counts()
        # Convert to DataFrame with proper column names to avoid shaking
        target_df = pd.DataFrame({
            'Class': target_counts.index.astype(str),
            'Count': target_counts.values
        })
        st.dataframe(target_df, hide_index=True, width=350)

    with col2:
        fig = px.bar(
            x=target_counts.index.astype(str),
            y=target_counts.values,
            labels={'x': 'Class', 'y': 'Count'},
            title='Target Class Distribution',
            color=target_counts.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Check if pre-trained models exist for sample dataset
    import os
    pretrained_available = False
    if data_source == "Use Sample Dataset":
        model_dir = 'model/saved_models'
        if os.path.exists(model_dir):
            pretrained_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
            if len(pretrained_files) >= 6:
                pretrained_available = True
                st.info("ℹ️ **Pre-trained models detected!** You can load them instantly (⚡ Fast) or train new models from scratch.")

    # Show option to load pre-trained models or train new ones
    if len(selected_models) > 0:
        col1, col2 = st.columns(2)

        with col1:
            if pretrained_available:
                if st.button("⚡ Load Pre-trained Models (Fast)", type="primary", use_container_width=True):
                    with st.spinner("📂 Loading pre-trained models..."):
                        try:
                            # Preprocess data for evaluation
                            X_train, X_test, y_train, y_test, feature_names, label_encoder = preprocess_data(
                                df, target_column, test_size=test_size/100, random_state=42  # Use same seed as training
                            )

                            # Initialize trainer
                            trainer = ModelTrainer(random_state=42)

                            # Load pre-trained models
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            for idx, model_name in enumerate(selected_models):
                                status_text.text(f"Loading {model_name}...")
                                clean_name = model_name.lower().replace(' ', '_')
                                filepath = os.path.join('model/saved_models', f'{clean_name}.pkl')

                                if os.path.exists(filepath):
                                    trainer.load_model(model_name, filepath)
                                    trainer.evaluate_model(model_name, X_test, y_test)
                                else:
                                    st.warning(f"⚠️ Pre-trained model not found for {model_name}, training from scratch...")
                                    trainer.train_model(model_name, X_train, y_train)
                                    trainer.evaluate_model(model_name, X_test, y_test)

                                progress_bar.progress((idx + 1) / len(selected_models))

                            status_text.text("✅ All models loaded successfully!")
                            progress_bar.empty()
                            status_text.empty()

                            # Store in session state
                            st.session_state.models_trained = True
                            st.session_state.trainer = trainer
                            st.session_state.X_test = X_test
                            st.session_state.y_test = y_test
                            st.session_state.label_encoder = label_encoder

                            st.success("✅ Pre-trained models loaded and evaluated successfully!")
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Error loading models: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

        with col2:
            if st.button("🚀 Train New Models", type="secondary", use_container_width=True):
                with st.spinner("🔄 Training models... This may take a few moments..."):
                    try:
                        # Preprocess data
                        X_train, X_test, y_train, y_test, feature_names, label_encoder = preprocess_data(
                            df, target_column, test_size=test_size/100, random_state=random_seed
                        )

                        # Initialize trainer
                        trainer = ModelTrainer(random_state=random_seed)

                        # Train only selected models
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for idx, model_name in enumerate(selected_models):
                            status_text.text(f"Training {model_name}...")
                            trainer.train_model(model_name, X_train, y_train)
                            trainer.evaluate_model(model_name, X_test, y_test)
                            progress_bar.progress((idx + 1) / len(selected_models))

                        status_text.text("✅ All models trained successfully!")
                        progress_bar.empty()
                        status_text.empty()

                        # Store in session state
                        st.session_state.models_trained = True
                        st.session_state.trainer = trainer
                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test
                        st.session_state.label_encoder = label_encoder

                        st.success("✅ All models trained and evaluated successfully!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error during training: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
    else:
        st.warning("⚠️ Please select at least one model to train.")

# Display results
if st.session_state.models_trained:
    trainer = st.session_state.trainer

    st.markdown("---")
    st.markdown('<h2 class="sub-header">📊 Model Performance Comparison</h2>', unsafe_allow_html=True)

    # Results table
    results_df = trainer.get_results_dataframe()
    st.markdown("### 📋 Evaluation Metrics Table")
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    # Download results
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="model_comparison_results.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # Visualizations
    st.markdown("### 📈 Performance Visualizations")

    # Prepare data for visualization
    metrics_for_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    plot_data = []
    for _, row in results_df.iterrows():
        for metric in metrics_for_plot:
            plot_data.append({
                'Model': row['ML Model Name'],
                'Metric': metric,
                'Score': float(row[metric])
            })
    plot_df = pd.DataFrame(plot_data)

    # Grouped bar chart
    fig = px.bar(
        plot_df,
        x='Model',
        y='Score',
        color='Metric',
        barmode='group',
        title='Model Performance Across All Metrics',
        height=500
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    st.markdown("### 🎯 Radar Chart Comparison")

    fig = go.Figure()

    for _, row in results_df.iterrows():
        model_name = row['ML Model Name']
        values = [float(row[metric]) for metric in metrics_for_plot]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics_for_plot,
            fill='toself',
            name=model_name
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Model Performance Radar Chart",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Confusion Matrix
    st.markdown("### 🔲 Confusion Matrix")

    # Create two columns: one for dropdown, one for chart
    col1, col2 = st.columns([1, 2])

    with col1:
        selected_model_cm = st.selectbox(
            "Select model for confusion matrix:",
            selected_models,
            key='cm_model'
        )

        # Display metrics for selected model
        if selected_model_cm in trainer.results:
            st.markdown("#### 📊 Evaluation Metrics")
            model_metrics = results_df[results_df['ML Model Name'] == selected_model_cm].iloc[0]

            # Create a clean metrics table
            metrics_dict = {
                'Metric': ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                'Score': [
                    f"{float(model_metrics['Accuracy']):.4f}",
                    f"{float(model_metrics['AUC']):.4f}",
                    f"{float(model_metrics['Precision']):.4f}",
                    f"{float(model_metrics['Recall']):.4f}",
                    f"{float(model_metrics['F1']):.4f}",
                    f"{float(model_metrics['MCC']):.4f}"
                ]
            }
            metrics_table_df = pd.DataFrame(metrics_dict)
            st.dataframe(metrics_table_df, hide_index=True, use_container_width=True)

    with col2:
        if selected_model_cm in trainer.results:
            cm = trainer.results[selected_model_cm]['confusion_matrix']

            # Clear any previous plots
            plt.clf()
            plt.close('all')

            # Create figure for confusion matrix
            fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 10},
                        square=True)
            ax.set_xlabel('Predicted Label', fontsize=10)
            ax.set_ylabel('True Label', fontsize=10)
            ax.set_title(f'Confusion Matrix - {selected_model_cm}', fontsize=11, fontweight='bold')
            ax.tick_params(labelsize=9)
            plt.tight_layout()

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.markdown("---")

    # Model Observations
    st.markdown("### 💡 Model Observations")

    observations = trainer.get_model_observations()

    obs_data = []
    for model_name in selected_models:
        if model_name in observations:
            obs_data.append({
                'ML Model Name': model_name,
                'Observation about model performance': observations[model_name]
            })

    obs_df = pd.DataFrame(obs_data)

    # Display observations as styled HTML table for better readability
    st.markdown("""
    <style>
    .obs-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
    }
    .obs-table th {
        padding: 12px;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid #444;
    }
    .obs-table td {
        padding: 15px;
        border-bottom: 1px solid #444;
        vertical-align: top;
        line-height: 1.6;
    }
    .model-name {
        width: 20%;
        font-weight: 500;
    }
    .observation {
        width: 80%;
        white-space: normal;
        word-wrap: break-word;
    }
    </style>
    """, unsafe_allow_html=True)

    # Build HTML table
    table_html = '<table class="obs-table"><thead><tr><th class="model-name">Model</th><th class="observation">Observations</th></tr></thead><tbody>'

    for _, row in obs_df.iterrows():
        table_html += f'<tr><td class="model-name">{row["ML Model Name"]}</td><td class="observation">{row["Observation about model performance"]}</td></tr>'

    table_html += '</tbody></table>'

    st.markdown(table_html, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>ML Classification Model Comparison Dashboard</strong></p>
    <p>Built with Streamlit 🎈 | scikit-learn | XGBoost</p>
</div>
""", unsafe_allow_html=True)


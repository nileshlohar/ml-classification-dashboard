# Machine Learning Classification Models Comparison

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ml-classification-dashboard.streamlit.app/)

## 🎯 Problem Statement

This project implements and compares six different classification algorithms to analyze their performance on a dataset with multiple features. The goal is to:
- Build a comprehensive machine learning pipeline
- Compare multiple classification models
- Evaluate models using standard metrics
- Create an interactive web application for demonstration
- Deploy the solution to the cloud

The project demonstrates an end-to-end ML workflow including data preprocessing, model training, evaluation, visualization, and deployment.

---

## 📊 Dataset Description

### Dataset Overview
- **Dataset Name**: Spambase Dataset
- **Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/ml/datasets/spambase
- **Type**: Binary Classification (Spam Detection)
- **Number of Instances**: 4,601 email messages
- **Number of Features**: 57 attributes
- **Number of Classes**: 2 (Spam=1, Not Spam=0)
- **Train-Test Split**: 80-20 (approximately 3,680 training, 921 testing instances)

### Dataset Characteristics
- **Domain**: Email spam detection and filtering
- **Total Features**: 57 numerical features
- **Feature Types**: All continuous real-valued features (percentages and counts)
- **Missing Values**: None
- **Class Distribution**: Approximately 39% spam, 61% non-spam (relatively balanced)
- **Feature Scaling**: StandardScaler applied to all features

### Features Description

The dataset contains 57 features derived from email content analysis:

#### Word Frequency Features (48 features)
Percentage of words in the email matching specific keywords:
- Business/commercial terms: make, address, all, business, email, you, your, credit, money, free
- Technical terms: internet, mail, receive, data, technology, hp, hpl, labs
- Common spam words: remove, order, people, report, addresses, font, 000
- Specific identifiers: george, 650, lab, 857, 415, 85, 1999, cs, edu
- And 28 more keyword frequencies

#### Character Frequency Features (6 features)
Percentage of specific characters in the email:
- `char_freq_semicolon` (;)
- `char_freq_parenthesis` (())
- `char_freq_bracket` ([])
- `char_freq_exclamation` (!)
- `char_freq_dollar` ($)
- `char_freq_hash` (#)

#### Capital Letter Features (3 features)
Statistics about sequences of consecutive capital letters:
- `capital_run_length_average`: Average length of uninterrupted capital letter sequences
- `capital_run_length_longest`: Length of longest uninterrupted capital letter sequence
- `capital_run_length_total`: Total number of capital letters in the email

### Target Variable
- **Name**: `target`
- **Type**: Binary (Categorical)
- **Classes**: 
  - **0**: Legitimate email (not spam) - 61% of dataset
  - **1**: Spam email - 39% of dataset
- **Distribution**: Reasonably balanced for binary classification

### Dataset Citation
```
Hopkins, M., Reeber, E., Forman, G., & Suermondt, J.
Spambase Data Set.
UCI Machine Learning Repository, 1999.
Hewlett-Packard Labs, 1501 Page Mill Rd., Palo Alto, CA 94304
```

### Use Case
This dataset is widely used for:
- Email spam filtering and detection
- Text classification research
- Feature engineering in NLP
- Binary classification benchmarking
- Machine learning algorithm comparison

### Why This Dataset?
✅ **Real-world application**: Actual email spam detection problem  
✅ **Rich feature set**: 57 features provide comprehensive email characteristics  
✅ **Sufficient size**: 4,601 instances for robust model training  
✅ **Less common**: Not as overused as Iris or Wine datasets  
✅ **Binary classification**: Clear, practical classification task  
✅ **Well-balanced**: Reasonable spam/non-spam distribution  


---

## 🤖 Models Used

### Comparison Table - Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.9294 | 0.9702 | 0.9293 | 0.9294 | 0.9293 | 0.8518 |
| Decision Tree | 0.9088 | 0.9097 | 0.9086 | 0.9088 | 0.9084 | 0.8081 |
| KNN | 0.9077 | 0.9506 | 0.9076 | 0.9077 | 0.9076 | 0.8065 |
| Naive Bayes | 0.8328 | 0.9376 | 0.8666 | 0.8328 | 0.8345 | 0.6946 |
| Random Forest (Ensemble) | 0.9381 | 0.9835 | 0.9387 | 0.9381 | 0.9377 | 0.8703 |
| XGBoost (Ensemble) | 0.9457 | 0.9880 | 0.9457 | 0.9457 | 0.9457 | 0.8863 |

> **Note**: These metrics are based on the Spambase dataset (4,601 instances, 57 features) with an 80-20 train-test split. Results may vary slightly with different random seeds or datasets.

### Metrics Explanation

1. **Accuracy**: Overall correctness of the model (correct predictions / total predictions)
2. **AUC (Area Under ROC Curve)**: Model's ability to distinguish between classes
3. **Precision**: Proportion of positive predictions that are actually correct
4. **Recall**: Proportion of actual positives that are correctly identified
5. **F1 Score**: Harmonic mean of Precision and Recall
6. **MCC (Matthews Correlation Coefficient)**: Balanced measure considering all confusion matrix elements

---

## 💡 Model Observations

### Performance Analysis for Each Model

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Linear model performing well for linearly separable data. Fast training and interpretable coefficients. May underfit complex non-linear patterns. Performance on this dataset: Moderate. Achieved 92.94% accuracy with excellent AUC (0.9702), making it a solid baseline model for spam detection. |
| Decision Tree | Captures non-linear relationships effectively. Prone to overfitting without proper pruning. Highly interpretable with feature importance. Performance on this dataset: Moderate. Achieved 90.88% accuracy, providing clear decision rules that can be easily understood and visualized. |
| KNN | Non-parametric lazy learner. Performance depends on distance metric and K value. Can be slow on large datasets. Sensitive to feature scaling. Performance on this dataset: Moderate. Achieved 90.77% accuracy with good AUC (0.9506), demonstrating the importance of proper feature scaling in the preprocessing step. |
| Naive Bayes | Probabilistic classifier assuming feature independence. Fast and efficient. Works well with small datasets. May underperform if independence assumption violated. Performance on this dataset: Moderate. Despite achieving lower accuracy (83.28%), it maintains good AUC (0.9376) and offers very fast training and prediction times. |
| Random Forest (Ensemble) | Ensemble method reducing overfitting through bagging. Robust and generally high accuracy. Provides feature importance. Less interpretable than single trees. Performance on this dataset: Moderate. Achieved strong performance with 93.81% accuracy and excellent AUC (0.9835), demonstrating the power of ensemble methods for this task. |
| XGBoost (Ensemble) | Gradient boosting ensemble with regularization. Often achieves highest accuracy. Handles missing values well. Requires careful hyperparameter tuning. Performance on this dataset: High. Top performer with 94.57% accuracy and outstanding AUC (0.9880), showcasing its effectiveness for structured spam detection data. |

### Key Insights

1. **Best Overall Performance**: XGBoost (94.57% accuracy, 0.8863 MCC)
2. **Most Balanced**: Random Forest (93.81% accuracy with robust metrics across the board)
3. **Fastest Training**: Naive Bayes and Logistic Regression
4. **Most Interpretable**: Decision Tree and Logistic Regression
5. **Best for Real-time Predictions**: Logistic Regression and Naive Bayes (faster prediction)
6. **Best for Complex Patterns**: XGBoost and Random Forest (ensemble methods)
7. **Lowest Performance**: Naive Bayes (83.28% accuracy) - likely due to feature independence assumption violations

### Recommendations

- **For Production**: Use XGBoost or Random Forest for best accuracy
- **For Interpretability**: Use Logistic Regression or Decision Tree
- **For Real-time Predictions**: Use Logistic Regression or Naive Bayes
- **For Maximum Accuracy**: Use XGBoost with hyperparameter tuning

---

## 🚀 Features

### Web Application Features

✅ **Dataset Upload**
- Support for CSV and Excel files
- Built-in sample dataset (Spambase) for quick testing
- Real-time dataset validation
- Display of dataset statistics and distribution

✅ **Model Selection**
- Multi-select dropdown to choose single or multiple models
- Train all 6 models simultaneously or individually
- **⚡ Pre-trained model loading** - Instantly load pre-trained models (sample dataset only)
- Progress tracking during training

✅ **Evaluation Metrics**
- Comprehensive metrics table with 6 key metrics per model
- Downloadable results in CSV format
- Real-time metric calculations

✅ **Visualization**
- Interactive confusion matrices for each model
- Grouped bar charts for metric comparison
- Radar charts for holistic performance view
- Performance heatmaps
- Target distribution visualization

✅ **Model Observations**
- Detailed performance analysis for each model
- Multi-line readable table format
- Insights and recommendations

✅ **User Controls**
- Adjustable train-test split ratio (default: 80-20)
- Random seed configuration for reproducibility
- Responsive and intuitive UI
- Dark/Light mode compatible

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/nileshlohar/ml-classification-dashboard.git
cd ml-classification-dashboard
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv

# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app**
```bash
streamlit run app.py
```

5. **Open in browser**
The app will automatically open at `http://localhost:8501`

---

### Live Application
🔗 **Live Demo**: [https://ml-classification-dashboard.streamlit.app/]

---

## 📖 Usage Guide

### Using the Web Application

1. **Select Data Source**
   - Choose "Upload CSV File" to use your own dataset
   - Choose "Use Sample Dataset" for quick demonstration

2. **Upload Your Dataset** (if using custom data)
   - Click "Browse files" 
   - Select a CSV or Excel file
   - Dataset must have minimum 500 instances and 12 features

3. **Select Target Column**
   - From the dropdown, choose the column you want to predict
   - View target distribution visualization

4. **Configure Training**
   - Select models to train (one or all)
   - Adjust train-test split ratio (default: 80-20)
   - Set random seed for reproducibility

5. **Train or Load Models**
   - **Option A**: Click "⚡ Load Pre-trained Models (Fast)" - Instantly loads pre-trained models (sample dataset only)
   - **Option B**: Click "🚀 Train New Models" - Trains models from scratch
   - Wait for completion and view progress bar

6. **Analyze Results**
   - Compare metrics in the table
   - Explore interactive visualizations
   - View confusion matrices
   - Read model observations
   - Download results as CSV

### Using the Training Script

1. **Run the standalone training script**
```bash
python train.py
```

2. **This will**:
   - Load and explore the sample dataset
   - Train all 6 models
   - Generate evaluation metrics
   - Create visualizations
   - Save results to CSV
   - Save trained models to `model/saved_models/`

---

## 🔧 Technologies Used

- **Python 3.9+**: Core programming language
- **Streamlit**: Web application framework
- **scikit-learn**: Machine learning library
- **XGBoost**: Gradient boosting framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical visualization
- **Plotly**: Interactive visualizations

---

## 🎓 About This Project

**Focus**: Machine Learning Classification Models  
**Purpose**: Interactive model comparison and evaluation dashboard  
**Implementation**: 6 classification algorithms with comprehensive metrics

---

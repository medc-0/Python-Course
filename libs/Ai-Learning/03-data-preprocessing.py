"""
03-data-preprocessing.py

Data Preprocessing for AI/ML
----------------------------
Learn essential data preprocessing techniques: scaling, encoding, feature
selection, and handling outliers. These steps are crucial for model performance.

What you'll learn
-----------------
1) Feature scaling (Standardization, Normalization)
2) Categorical encoding (One-hot, Label encoding)
3) Feature selection techniques
4) Outlier detection and handling
5) Data splitting for training/validation

Key Libraries
-------------
- sklearn.preprocessing: Scaling and encoding
- sklearn.feature_selection: Feature selection methods
- sklearn.model_selection: Data splitting and validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def demonstrate_feature_scaling():
    """Show different feature scaling techniques"""
    print("=== Feature Scaling Techniques ===")
    
    # Create sample data with different scales
    np.random.seed(42)
    data = {
        'age': np.random.randint(18, 80, 100),
        'income': np.random.normal(50000, 15000, 100),
        'height': np.random.normal(170, 10, 100),
        'weight': np.random.normal(70, 15, 100)
    }
    df = pd.DataFrame(data)
    
    print("Original data statistics:")
    print(df.describe())
    
    # 1. Standardization (Z-score normalization)
    scaler_std = StandardScaler()
    df_standardized = pd.DataFrame(
        scaler_std.fit_transform(df), 
        columns=df.columns
    )
    print(f"\nAfter Standardization (mean=0, std=1):")
    print(df_standardized.describe())
    
    # 2. Min-Max Scaling (0-1 range)
    scaler_minmax = MinMaxScaler()
    df_minmax = pd.DataFrame(
        scaler_minmax.fit_transform(df), 
        columns=df.columns
    )
    print(f"\nAfter Min-Max Scaling (0-1 range):")
    print(df_minmax.describe())
    
    # Visualize scaling effects
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Feature Scaling Comparison', fontsize=16)
    
    for i, col in enumerate(['age', 'income']):
        # Original data
        axes[0, i].hist(df[col], bins=20, alpha=0.7, color='blue', label='Original')
        axes[0, i].set_title(f'{col} - Original')
        axes[0, i].set_xlabel(col)
        axes[0, i].set_ylabel('Frequency')
        
        # Standardized data
        axes[1, i].hist(df_standardized[col], bins=20, alpha=0.7, color='red', label='Standardized')
        axes[1, i].set_title(f'{col} - Standardized')
        axes[1, i].set_xlabel(col)
        axes[1, i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('feature_scaling.png', dpi=300, bbox_inches='tight')
    print("Scaling visualization saved as 'feature_scaling.png'")
    plt.show()
    
    return df, df_standardized, df_minmax

def categorical_encoding():
    """Demonstrate categorical variable encoding"""
    print("\n=== Categorical Encoding ===")
    
    # Create sample categorical data
    data = {
        'city': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney'] * 20,
        'category': ['A', 'B', 'C'] * 33 + ['A'],
        'size': ['Small', 'Medium', 'Large'] * 33 + ['Small'],
        'target': np.random.choice([0, 1], 100)
    }
    df = pd.DataFrame(data)
    
    print("Original categorical data:")
    print(df.head(10))
    print(f"\nUnique values:")
    for col in ['city', 'category', 'size']:
        print(f"{col}: {df[col].nunique()} unique values")
    
    # 1. Label Encoding (for ordinal data)
    label_encoder = LabelEncoder()
    df_labeled = df.copy()
    df_labeled['size_encoded'] = label_encoder.fit_transform(df['size'])
    
    print(f"\nLabel Encoding for 'size':")
    print(f"Mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    print(df_labeled[['size', 'size_encoded']].head())
    
    # 2. One-Hot Encoding (for nominal data)
    df_onehot = pd.get_dummies(df, columns=['city', 'category'], prefix=['city', 'cat'])
    print(f"\nOne-Hot Encoding:")
    print(f"Original shape: {df.shape}")
    print(f"After one-hot: {df_onehot.shape}")
    print(f"New columns: {[col for col in df_onehot.columns if col.startswith(('city_', 'cat_'))]}")
    
    # 3. Target encoding (mean encoding)
    df_target = df.copy()
    for col in ['city', 'category']:
        target_mean = df.groupby(col)['target'].mean()
        df_target[f'{col}_target_encoded'] = df[col].map(target_mean)
    
    print(f"\nTarget Encoding:")
    print(df_target[['city', 'city_target_encoded', 'category', 'category_target_encoded']].head())
    
    return df, df_labeled, df_onehot, df_target

def feature_selection():
    """Demonstrate feature selection techniques"""
    print("\n=== Feature Selection ===")
    
    # Create dataset with informative and redundant features
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, 
        n_redundant=5, n_clusters_per_class=1, random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")
    
    # 1. Statistical feature selection
    selector_f = SelectKBest(score_func=f_classif, k=10)
    X_selected_f = selector_f.fit_transform(X, y)
    selected_features_f = [feature_names[i] for i in selector_f.get_support(indices=True)]
    
    print(f"\nF-test feature selection (top 10):")
    print(f"Selected features: {selected_features_f}")
    print(f"Scores: {selector_f.scores_[selector_f.get_support()]}")
    
    # 2. Mutual information feature selection
    selector_mi = SelectKBest(score_func=mutual_info_classif, k=10)
    X_selected_mi = selector_mi.fit_transform(X, y)
    selected_features_mi = [feature_names[i] for i in selector_mi.get_support(indices=True)]
    
    print(f"\nMutual Information feature selection (top 10):")
    print(f"Selected features: {selected_features_mi}")
    print(f"Scores: {selector_mi.scores_[selector_mi.get_support()]}")
    
    # Visualize feature importance
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # F-test scores
    f_scores = selector_f.scores_
    axes[0].bar(range(len(f_scores)), f_scores)
    axes[0].set_title('F-test Feature Scores')
    axes[0].set_xlabel('Feature Index')
    axes[0].set_ylabel('F-score')
    axes[0].axhline(y=np.mean(f_scores), color='r', linestyle='--', label='Mean')
    axes[0].legend()
    
    # Mutual information scores
    mi_scores = selector_mi.scores_
    axes[1].bar(range(len(mi_scores)), mi_scores)
    axes[1].set_title('Mutual Information Feature Scores')
    axes[1].set_xlabel('Feature Index')
    axes[1].set_ylabel('MI Score')
    axes[1].axhline(y=np.mean(mi_scores), color='r', linestyle='--', label='Mean')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('feature_selection.png', dpi=300, bbox_inches='tight')
    print("Feature selection visualization saved as 'feature_selection.png'")
    plt.show()
    
    return X, y, selected_features_f, selected_features_mi

def outlier_detection():
    """Demonstrate outlier detection and handling"""
    print("\n=== Outlier Detection ===")
    
    # Create data with outliers
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 1000)
    outliers = np.random.normal(0, 1, 50) * 5  # Create outliers
    data_with_outliers = np.concatenate([normal_data, outliers])
    
    print(f"Data shape: {data_with_outliers.shape}")
    print(f"Mean: {data_with_outliers.mean():.2f}")
    print(f"Std: {data_with_outliers.std():.2f}")
    
    # 1. Statistical outlier detection (Z-score)
    z_scores = np.abs((data_with_outliers - data_with_outliers.mean()) / data_with_outliers.std())
    outliers_z = z_scores > 3
    print(f"\nZ-score method detected {outliers_z.sum()} outliers")
    
    # 2. IQR method
    Q1 = np.percentile(data_with_outliers, 25)
    Q3 = np.percentile(data_with_outliers, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_iqr = (data_with_outliers < lower_bound) | (data_with_outliers > upper_bound)
    print(f"IQR method detected {outliers_iqr.sum()} outliers")
    
    # 3. Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outlier_labels = iso_forest.fit_predict(data_with_outliers.reshape(-1, 1))
    outliers_iso = outlier_labels == -1
    print(f"Isolation Forest detected {outliers_iso.sum()} outliers")
    
    # Visualize outliers
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Outlier Detection Methods', fontsize=16)
    
    # Original data
    axes[0, 0].hist(data_with_outliers, bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('Original Data')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # Z-score outliers
    axes[0, 1].scatter(range(len(data_with_outliers)), data_with_outliers, 
                      c=outliers_z, cmap='RdYlBu', alpha=0.7)
    axes[0, 1].set_title('Z-score Outliers (Red)')
    axes[0, 1].set_xlabel('Index')
    axes[0, 1].set_ylabel('Value')
    
    # IQR outliers
    axes[1, 0].scatter(range(len(data_with_outliers)), data_with_outliers, 
                      c=outliers_iqr, cmap='RdYlBu', alpha=0.7)
    axes[1, 0].set_title('IQR Outliers (Red)')
    axes[1, 0].set_xlabel('Index')
    axes[1, 0].set_ylabel('Value')
    
    # Isolation Forest outliers
    axes[1, 1].scatter(range(len(data_with_outliers)), data_with_outliers, 
                      c=outliers_iso, cmap='RdYlBu', alpha=0.7)
    axes[1, 1].set_title('Isolation Forest Outliers (Red)')
    axes[1, 1].set_xlabel('Index')
    axes[1, 1].set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig('outlier_detection.png', dpi=300, bbox_inches='tight')
    print("Outlier detection visualization saved as 'outlier_detection.png'")
    plt.show()
    
    return data_with_outliers, outliers_z, outliers_iqr, outliers_iso

def data_splitting():
    """Demonstrate proper data splitting for ML"""
    print("\n=== Data Splitting for ML ===")
    
    # Create sample dataset
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
    
    # Basic train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Original dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/X.shape[0]*100:.1f}%)")
    
    # Check class distribution
    print(f"\nClass distribution:")
    print(f"Original: {np.bincount(y)}")
    print(f"Training: {np.bincount(y_train)}")
    print(f"Test: {np.bincount(y_test)}")
    
    # Further split training into train and validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\nAfter validation split:")
    print(f"Final training: {X_train_final.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    
    return X_train_final, X_val, X_test, y_train_final, y_val, y_test

def complete_preprocessing_pipeline():
    """Demonstrate a complete preprocessing pipeline"""
    print("\n=== Complete Preprocessing Pipeline ===")
    
    # Create mixed dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Numerical features
    age = np.random.randint(18, 80, n_samples)
    income = np.random.normal(50000, 15000, n_samples)
    
    # Categorical features
    city = np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n_samples)
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
    
    # Target variable
    target = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'city': city,
        'education': education,
        'target': target
    })
    
    print("Original dataset:")
    print(df.head())
    print(f"\nDataset info:")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # 1. Handle categorical variables
    df_processed = df.copy()
    
    # Label encode education (ordinal)
    education_order = ['High School', 'Bachelor', 'Master', 'PhD']
    df_processed['education_encoded'] = df_processed['education'].map(
        {edu: i for i, edu in enumerate(education_order)}
    )
    
    # One-hot encode city (nominal)
    df_processed = pd.get_dummies(df_processed, columns=['city'], prefix='city')
    
    # 2. Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['age', 'income']
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    
    # 3. Split data
    X = df_processed.drop(['target', 'education'], axis=1)
    y = df_processed['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nAfter preprocessing:")
    print(f"Features: {list(X.columns)}")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Sample of processed data:\n{X_train.head()}")
    
    return X_train, X_test, y_train, y_test, scaler

def main():
    """Main function to run all preprocessing demonstrations"""
    print("🔧 Data Preprocessing for AI/ML")
    print("=" * 50)
    
    # Feature scaling
    demonstrate_feature_scaling()
    
    # Categorical encoding
    categorical_encoding()
    
    # Feature selection
    feature_selection()
    
    # Outlier detection
    outlier_detection()
    
    # Data splitting
    data_splitting()
    
    # Complete pipeline
    complete_preprocessing_pipeline()
    
    print("\n" + "=" * 50)
    print("✅ Lesson 3 Complete!")
    print("Next: Learn about machine learning algorithms with scikit-learn")
    print("Key takeaway: Proper preprocessing is crucial for model performance")

if __name__ == "__main__":
    main()

"""
04-data-visualization.py

Data Visualization for AI/ML
----------------------------
Master data visualization techniques essential for AI/ML: exploratory data
analysis, feature relationships, model performance visualization, and
creating publication-ready plots.

What you'll learn
-----------------
1) Matplotlib and Seaborn for statistical visualization
2) Exploratory data analysis (EDA) techniques
3) Feature correlation and distribution analysis
4) Model performance visualization
5) Interactive visualizations with Plotly

Key Libraries
-------------
- matplotlib: Basic plotting and customization
- seaborn: Statistical data visualization
- plotly: Interactive visualizations
- pandas: Built-in plotting capabilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_iris, load_wine, make_classification
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def basic_plotting_techniques():
    """Demonstrate basic plotting techniques"""
    print("=== Basic Plotting Techniques ===")
    
    # Create sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x) * np.cos(x)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Basic Plotting Techniques', fontsize=16)
    
    # Line plot
    axes[0, 0].plot(x, y1, label='sin(x)', color='blue', linewidth=2)
    axes[0, 0].plot(x, y2, label='cos(x)', color='red', linewidth=2)
    axes[0, 0].set_title('Line Plot')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Scatter plot
    np.random.seed(42)
    x_scatter = np.random.randn(100)
    y_scatter = 2 * x_scatter + np.random.randn(100) * 0.5
    axes[0, 1].scatter(x_scatter, y_scatter, alpha=0.6, color='green', s=50)
    axes[0, 1].set_title('Scatter Plot')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histogram
    data_hist = np.random.normal(0, 1, 1000)
    axes[1, 0].hist(data_hist, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_title('Histogram')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bar plot
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 45, 56, 78, 32]
    axes[1, 1].bar(categories, values, color='purple', alpha=0.7)
    axes[1, 1].set_title('Bar Plot')
    axes[1, 1].set_xlabel('Category')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('basic_plotting.png', dpi=300, bbox_inches='tight')
    print("Basic plotting saved as 'basic_plotting.png'")
    plt.show()

def exploratory_data_analysis():
    """Demonstrate comprehensive EDA techniques"""
    print("\n=== Exploratory Data Analysis ===")
    
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("Dataset overview:")
    print(f"Shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    print(f"Species distribution:\n{df['species_name'].value_counts()}")
    
    # Create comprehensive EDA visualization
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Pair plot (correlation matrix)
    plt.subplot(2, 3, 1)
    numeric_cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Feature Correlation Matrix')
    
    # 2. Distribution of each feature
    plt.subplot(2, 3, 2)
    df[numeric_cols].hist(bins=20, alpha=0.7, figsize=(8, 6))
    plt.title('Feature Distributions')
    plt.tight_layout()
    
    # 3. Box plots by species
    plt.subplot(2, 3, 3)
    df_melted = df.melt(id_vars=['species_name'], value_vars=numeric_cols, 
                       var_name='feature', value_name='value')
    sns.boxplot(data=df_melted, x='feature', y='value', hue='species_name')
    plt.title('Feature Distribution by Species')
    plt.xticks(rotation=45)
    
    # 4. Scatter plot matrix
    plt.subplot(2, 3, 4)
    colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
    for species in colors:
        data = df[df['species_name'] == species]
        plt.scatter(data['sepal length (cm)'], data['petal length (cm)'], 
                   c=colors[species], label=species, alpha=0.7)
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.title('Sepal vs Petal Length')
    plt.legend()
    
    # 5. Violin plots
    plt.subplot(2, 3, 5)
    sns.violinplot(data=df, x='species_name', y='petal width (cm)')
    plt.title('Petal Width Distribution by Species')
    
    # 6. Feature importance (using variance)
    plt.subplot(2, 3, 6)
    feature_variance = df[numeric_cols].var()
    plt.bar(range(len(feature_variance)), feature_variance.values)
    plt.xticks(range(len(feature_variance)), feature_variance.index, rotation=45)
    plt.title('Feature Variance')
    plt.ylabel('Variance')
    
    plt.tight_layout()
    plt.savefig('eda_comprehensive.png', dpi=300, bbox_inches='tight')
    print("EDA visualization saved as 'eda_comprehensive.png'")
    plt.show()
    
    return df

def model_performance_visualization():
    """Demonstrate model performance visualization techniques"""
    print("\n=== Model Performance Visualization ===")
    
    # Create sample classification data
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, 
                              n_redundant=0, n_clusters_per_class=1, random_state=42)
    
    # Split data
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Create performance visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Performance Visualization', fontsize=16)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Decision Boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    axes[1, 0].contourf(xx, yy, Z, alpha=0.8, cmap='RdYlBu')
    scatter = axes[1, 0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', edgecolors='black')
    axes[1, 0].set_title('Decision Boundary')
    axes[1, 0].set_xlabel('Feature 1')
    axes[1, 0].set_ylabel('Feature 2')
    
    # 4. Learning Curve
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    axes[1, 1].plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    axes[1, 1].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    axes[1, 1].plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    axes[1, 1].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    axes[1, 1].set_title('Learning Curve')
    axes[1, 1].set_xlabel('Training Set Size')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    print("Model performance visualization saved as 'model_performance.png'")
    plt.show()

def interactive_visualizations():
    """Demonstrate interactive visualizations with Plotly"""
    print("\n=== Interactive Visualizations ===")
    
    # Load wine dataset
    wine = load_wine()
    df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
    df_wine['target'] = wine.target
    df_wine['wine_type'] = df_wine['target'].map({0: 'Class 0', 1: 'Class 1', 2: 'Class 2'})
    
    # Create interactive scatter plot
    fig = px.scatter(df_wine, x='alcohol', y='malic_acid', color='wine_type',
                     size='total_phenols', hover_data=['flavanoids'],
                     title='Interactive Wine Dataset Visualization')
    
    # Save as HTML
    fig.write_html('interactive_wine_plot.html')
    print("Interactive plot saved as 'interactive_wine_plot.html'")
    
    # Create 3D scatter plot
    fig_3d = px.scatter_3d(df_wine, x='alcohol', y='malic_acid', z='flavanoids',
                          color='wine_type', title='3D Wine Dataset Visualization')
    fig_3d.write_html('interactive_3d_plot.html')
    print("3D interactive plot saved as 'interactive_3d_plot.html'")
    
    # Create interactive correlation heatmap
    corr_matrix = df_wine.select_dtypes(include=[np.number]).corr()
    fig_heatmap = px.imshow(corr_matrix, title='Interactive Correlation Heatmap')
    fig_heatmap.write_html('interactive_heatmap.html')
    print("Interactive heatmap saved as 'interactive_heatmap.html'")
    
    return df_wine

def advanced_visualization_techniques():
    """Demonstrate advanced visualization techniques"""
    print("\n=== Advanced Visualization Techniques ===")
    
    # Create sample time series data
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    np.random.seed(42)
    trend = np.linspace(100, 200, 365)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 365.25)
    noise = np.random.normal(0, 5, 365)
    values = trend + seasonal + noise
    
    df_time = pd.DataFrame({'date': dates, 'value': values})
    
    # Create advanced visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Advanced Visualization Techniques', fontsize=16)
    
    # 1. Time series with trend
    axes[0, 0].plot(df_time['date'], df_time['value'], alpha=0.7, color='blue')
    # Add trend line
    z = np.polyfit(range(len(df_time)), df_time['value'], 1)
    p = np.poly1d(z)
    axes[0, 0].plot(df_time['date'], p(range(len(df_time))), "r--", alpha=0.8, linewidth=2)
    axes[0, 0].set_title('Time Series with Trend')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Multi-line plot with different metrics
    metrics = ['metric_a', 'metric_b', 'metric_c']
    data_multi = {}
    for i, metric in enumerate(metrics):
        data_multi[metric] = np.random.randn(100).cumsum() + i * 10
    
    for metric in metrics:
        axes[0, 1].plot(data_multi[metric], label=metric, linewidth=2)
    axes[0, 1].set_title('Multiple Metrics Comparison')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Stacked area chart
    categories = ['A', 'B', 'C', 'D']
    time_points = np.arange(1, 11)
    data_stacked = np.random.rand(4, 10) * 100
    
    axes[1, 0].stackplot(time_points, data_stacked, labels=categories, alpha=0.7)
    axes[1, 0].set_title('Stacked Area Chart')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend(loc='upper left')
    
    # 4. Radar chart
    categories_radar = ['Speed', 'Reliability', 'Comfort', 'Safety', 'Efficiency']
    values_radar = [8, 6, 7, 9, 5]
    
    angles = np.linspace(0, 2 * np.pi, len(categories_radar), endpoint=False).tolist()
    values_radar += values_radar[:1]  # Complete the circle
    angles += angles[:1]
    
    axes[1, 1].plot(angles, values_radar, 'o-', linewidth=2, color='purple')
    axes[1, 1].fill(angles, values_radar, alpha=0.25, color='purple')
    axes[1, 1].set_xticks(angles[:-1])
    axes[1, 1].set_xticklabels(categories_radar)
    axes[1, 1].set_ylim(0, 10)
    axes[1, 1].set_title('Radar Chart')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('advanced_visualizations.png', dpi=300, bbox_inches='tight')
    print("Advanced visualizations saved as 'advanced_visualizations.png'")
    plt.show()

def main():
    """Main function to run all visualization demonstrations"""
    print("📊 Data Visualization for AI/ML")
    print("=" * 50)
    
    # Basic plotting techniques
    basic_plotting_techniques()
    
    # Exploratory data analysis
    exploratory_data_analysis()
    
    # Model performance visualization
    model_performance_visualization()
    
    # Interactive visualizations
    interactive_visualizations()
    
    # Advanced techniques
    advanced_visualization_techniques()
    
    print("\n" + "=" * 50)
    print("✅ Lesson 4 Complete!")
    print("Next: Learn machine learning algorithms with scikit-learn")
    print("Key takeaway: Good visualizations reveal patterns and insights in data")

if __name__ == "__main__":
    main()

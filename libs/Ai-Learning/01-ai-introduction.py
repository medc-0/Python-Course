"""
01-ai-introduction.py

AI Introduction & Fundamentals
------------------------------
This lesson introduces artificial intelligence concepts, types of learning,
and sets up the environment for AI development.

What you'll learn
-----------------
1) What is AI, ML, and Deep Learning
2) Types of machine learning (supervised, unsupervised, reinforcement)
3) Common AI libraries and their purposes
4) Setting up a basic AI development environment

Key Libraries
-------------
- NumPy: Numerical computing foundation
- Pandas: Data manipulation and analysis
- Matplotlib: Data visualization
- Scikit-learn: Machine learning algorithms
- TensorFlow/PyTorch: Deep learning frameworks

AI Learning Types
-----------------
1. Supervised Learning: Learn from labeled data (classification, regression)
2. Unsupervised Learning: Find patterns in unlabeled data (clustering, dimensionality reduction)
3. Reinforcement Learning: Learn through interaction and rewards
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def demonstrate_data_types():
    """Show different types of data used in AI/ML"""
    print("=== AI Data Types Demo ===")
    
    # 1. Numerical data (continuous)
    heights = np.array([165, 170, 175, 180, 185, 190])
    weights = np.array([60, 65, 70, 75, 80, 85])
    print(f"Heights: {heights}")
    print(f"Weights: {weights}")
    
    # 2. Categorical data (discrete)
    categories = np.array(['cat', 'dog', 'bird', 'cat', 'dog'])
    print(f"Categories: {categories}")
    
    # 3. Text data
    texts = ["I love AI", "Machine learning is fascinating", "Deep learning rocks"]
    print(f"Text samples: {texts}")
    
    return heights, weights, categories, texts

def create_sample_datasets():
    """Generate sample datasets for different ML tasks"""
    print("\n=== Sample Datasets ===")
    
    # 1. Classification dataset (supervised learning)
    X_class, y_class = make_classification(
        n_samples=100, n_features=2, n_informative=2, 
        n_redundant=0, n_clusters_per_class=1, random_state=42
    )
    print(f"Classification data shape: {X_class.shape}")
    print(f"Classification labels: {np.unique(y_class)}")
    
    # 2. Regression dataset (supervised learning)
    X_reg, y_reg = make_regression(
        n_samples=100, n_features=1, noise=10, random_state=42
    )
    print(f"Regression data shape: {X_reg.shape}")
    print(f"Regression target range: {y_reg.min():.2f} to {y_reg.max():.2f}")
    
    # 3. Clustering dataset (unsupervised learning)
    X_cluster, _ = make_classification(
        n_samples=100, n_features=2, n_informative=2,
        n_redundant=0, n_clusters_per_class=1, random_state=42
    )
    print(f"Clustering data shape: {X_cluster.shape}")
    
    return X_class, y_class, X_reg, y_reg, X_cluster

def visualize_ai_concepts():
    """Create visualizations to explain AI concepts"""
    print("\n=== Creating AI Concept Visualizations ===")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('AI Learning Types Visualization', fontsize=16)
    
    # 1. Supervised Learning - Classification
    X_class, y_class, X_reg, y_reg, X_cluster = create_sample_datasets()
    
    # Classification plot
    scatter1 = axes[0, 0].scatter(X_class[:, 0], X_class[:, 1], c=y_class, cmap='viridis', alpha=0.7)
    axes[0, 0].set_title('Supervised Learning: Classification')
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')
    plt.colorbar(scatter1, ax=axes[0, 0])
    
    # Regression plot
    axes[0, 1].scatter(X_reg, y_reg, alpha=0.7, color='orange')
    axes[0, 1].set_title('Supervised Learning: Regression')
    axes[0, 1].set_xlabel('Feature')
    axes[0, 1].set_ylabel('Target')
    
    # Unsupervised Learning - Clustering
    axes[1, 0].scatter(X_cluster[:, 0], X_cluster[:, 1], alpha=0.7, color='green')
    axes[1, 0].set_title('Unsupervised Learning: Clustering')
    axes[1, 0].set_xlabel('Feature 1')
    axes[1, 0].set_ylabel('Feature 2')
    
    # AI Process Flow
    axes[1, 1].text(0.5, 0.8, 'Data Collection', ha='center', fontsize=12, weight='bold')
    axes[1, 1].text(0.5, 0.6, '↓', ha='center', fontsize=16)
    axes[1, 1].text(0.5, 0.5, 'Data Preprocessing', ha='center', fontsize=12, weight='bold')
    axes[1, 1].text(0.5, 0.3, '↓', ha='center', fontsize=16)
    axes[1, 1].text(0.5, 0.2, 'Model Training', ha='center', fontsize=12, weight='bold')
    axes[1, 1].text(0.5, 0.0, '↓', ha='center', fontsize=16)
    axes[1, 1].text(0.5, -0.1, 'Prediction/Inference', ha='center', fontsize=12, weight='bold')
    axes[1, 1].set_title('AI Process Flow')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(-0.2, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('ai_concepts.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'ai_concepts.png'")
    plt.show()

def demonstrate_basic_ml_pipeline():
    """Show a basic machine learning pipeline"""
    print("\n=== Basic ML Pipeline Demo ===")
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=4, 
                              n_redundant=0, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Basic statistics
    print(f"\nFeature statistics:")
    for i in range(X_train.shape[1]):
        print(f"Feature {i+1}: mean={X_train[:, i].mean():.2f}, std={X_train[:, i].std():.2f}")
    
    return X_train, X_test, y_train, y_test

def main():
    """Main function to run all demonstrations"""
    print("🤖 Welcome to AI Learning Course!")
    print("=" * 50)
    
    # Demonstrate different data types
    demonstrate_data_types()
    
    # Create sample datasets
    create_sample_datasets()
    
    # Visualize AI concepts
    visualize_ai_concepts()
    
    # Show basic ML pipeline
    demonstrate_basic_ml_pipeline()
    
    print("\n" + "=" * 50)
    print("✅ Lesson 1 Complete!")
    print("Next: Learn about data preprocessing and feature engineering")
    print("Key takeaway: AI is about finding patterns in data to make predictions")

if __name__ == "__main__":
    main()

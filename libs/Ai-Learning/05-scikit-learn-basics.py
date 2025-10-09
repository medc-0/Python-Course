"""
05-scikit-learn-basics.py

Scikit-learn Fundamentals
-------------------------
Learn the essential scikit-learn library for machine learning. This lesson
covers the core concepts, API design, and basic algorithms.

What you'll learn
-----------------
1) Scikit-learn API design and conventions
2) Basic supervised learning algorithms
3) Model training, prediction, and evaluation
4) Cross-validation and hyperparameter tuning
5) Pipeline creation for data preprocessing

Key Concepts
------------
- Estimator interface: fit(), predict(), transform()
- Transformer interface: fit_transform()
- Predictor interface: score()
- Cross-validation for model evaluation
- Grid search for hyperparameter optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def scikit_learn_api_overview():
    """Demonstrate the scikit-learn API design"""
    print("=== Scikit-learn API Overview ===")
    
    # Load sample data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target classes: {np.unique(y)}")
    print(f"Feature names: {iris.feature_names}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Demonstrate the estimator interface
    print("\n=== Estimator Interface ===")
    
    # 1. Create estimator
    clf = LogisticRegression(random_state=42)
    print(f"Estimator type: {type(clf)}")
    print(f"Available methods: {[method for method in dir(clf) if not method.startswith('_')]}")
    
    # 2. Fit the estimator
    clf.fit(X_train, y_train)
    print(f"Model fitted: {clf.__class__.__name__}")
    
    # 3. Make predictions
    y_pred = clf.predict(X_test)
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Sample predictions: {y_pred[:5]}")
    
    # 4. Evaluate the model
    accuracy = clf.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.3f}")
    
    return X_train, X_test, y_train, y_test, clf

def basic_supervised_algorithms():
    """Demonstrate basic supervised learning algorithms"""
    print("\n=== Basic Supervised Learning Algorithms ===")
    
    # Create classification dataset
    X_class, y_class = make_classification(n_samples=1000, n_features=4, n_informative=4,
                                          n_redundant=0, n_clusters_per_class=1, random_state=42)
    
    # Create regression dataset
    X_reg, y_reg = make_regression(n_samples=1000, n_features=4, noise=10, random_state=42)
    
    # Split both datasets
    X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
        X_class, y_class, test_size=0.3, random_state=42)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42)
    
    # Classification algorithms
    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    print("Classification Results:")
    print("-" * 50)
    for name, clf in classifiers.items():
        clf.fit(X_class_train, y_class_train)
        y_pred = clf.predict(X_class_test)
        accuracy = accuracy_score(y_class_test, y_pred)
        print(f"{name:20}: {accuracy:.3f}")
    
    # Regression algorithms
    regressors = {
        'Linear Regression': LinearRegression(),
        'Ridge': LinearRegression(),  # Simplified for demo
        'Lasso': LinearRegression()   # Simplified for demo
    }
    
    print("\nRegression Results:")
    print("-" * 50)
    for name, reg in regressors.items():
        reg.fit(X_reg_train, y_reg_train)
        y_pred = reg.predict(X_reg_test)
        mse = mean_squared_error(y_reg_test, y_pred)
        print(f"{name:20}: MSE = {mse:.2f}")
    
    return classifiers, regressors

def cross_validation_demo():
    """Demonstrate cross-validation techniques"""
    print("\n=== Cross-Validation Demo ===")
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Create different models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    print("Cross-Validation Results (5-fold):")
    print("-" * 50)
    
    cv_results = {}
    for name, model in models.items():
        # Perform 5-fold cross-validation
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_results[name] = scores
        
        print(f"{name:20}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        print(f"Individual scores: {scores}")
        print()
    
    # Visualize cross-validation results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Box plot of CV scores
    data_to_plot = [scores for scores in cv_results.values()]
    box_plot = ax.boxplot(data_to_plot, labels=list(cv_results.keys()), patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_title('Cross-Validation Score Distribution')
    ax.set_ylabel('Accuracy')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('cross_validation_results.png', dpi=300, bbox_inches='tight')
    print("Cross-validation visualization saved as 'cross_validation_results.png'")
    plt.show()
    
    return cv_results

def hyperparameter_tuning():
    """Demonstrate hyperparameter tuning with GridSearchCV"""
    print("\n=== Hyperparameter Tuning ===")
    
    # Load data
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=4,
                              n_redundant=0, random_state=42)
    
    # Define parameter grid for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print("Parameter grid for Random Forest:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # Create GridSearchCV object
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring='accuracy', 
        n_jobs=-1, verbose=1
    )
    
    # Fit the grid search
    print("\nPerforming grid search...")
    grid_search.fit(X, y)
    
    # Results
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    print(f"Best estimator: {grid_search.best_estimator_}")
    
    # Get results for visualization
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Plot parameter importance (simplified)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Hyperparameter Tuning Results', fontsize=16)
    
    # Plot mean test scores for different parameter values
    param_names = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
    
    for i, param in enumerate(param_names):
        ax = axes[i // 2, i % 2]
        
        # Group by parameter value and calculate mean score
        param_scores = results_df.groupby(f'param_{param}')['mean_test_score'].mean()
        
        ax.plot(param_scores.index, param_scores.values, 'o-', linewidth=2, markersize=8)
        ax.set_title(f'Mean CV Score vs {param}')
        ax.set_xlabel(param)
        ax.set_ylabel('Mean CV Score')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
    print("Hyperparameter tuning visualization saved as 'hyperparameter_tuning.png'")
    plt.show()
    
    return grid_search

def pipeline_creation():
    """Demonstrate creating ML pipelines"""
    print("\n=== ML Pipeline Creation ===")
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    print("Pipeline steps:")
    for i, (name, step) in enumerate(pipeline.steps):
        print(f"  {i+1}. {name}: {step.__class__.__name__}")
    
    # Fit and evaluate pipeline
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nPipeline accuracy: {accuracy:.3f}")
    
    # Compare with and without scaling
    print("\nComparison: With vs Without Scaling")
    print("-" * 40)
    
    # Without scaling
    clf_no_scale = LogisticRegression(random_state=42)
    clf_no_scale.fit(X_train, y_train)
    acc_no_scale = clf_no_scale.score(X_test, y_test)
    print(f"Without scaling: {acc_no_scale:.3f}")
    
    # With scaling
    acc_with_scale = pipeline.score(X_test, y_test)
    print(f"With scaling:    {acc_with_scale:.3f}")
    
    # Create more complex pipeline
    complex_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Use GridSearchCV with pipeline
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5, None]
    }
    
    grid_pipeline = GridSearchCV(complex_pipeline, param_grid, cv=3, scoring='accuracy')
    grid_pipeline.fit(X_train, y_train)
    
    print(f"\nBest pipeline score: {grid_pipeline.best_score_:.3f}")
    print(f"Best parameters: {grid_pipeline.best_params_}")
    
    return pipeline, grid_pipeline

def model_comparison():
    """Compare different models on the same dataset"""
    print("\n=== Model Comparison ===")
    
    # Load data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                              n_redundant=2, n_clusters_per_class=1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define models to compare
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM (Linear)': SVC(kernel='linear', random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            'test_accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"{name:20}: Test Acc = {accuracy:.3f}, CV = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Test accuracy comparison
    names = list(results.keys())
    test_accs = [results[name]['test_accuracy'] for name in names]
    cv_means = [results[name]['cv_mean'] for name in names]
    cv_stds = [results[name]['cv_std'] for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax1.bar(x - width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
    ax1.bar(x + width/2, cv_means, width, label='CV Mean', alpha=0.8)
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # CV scores with error bars
    ax2.bar(names, cv_means, yerr=cv_stds, capsize=5, alpha=0.8)
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Cross-Validation Accuracy')
    ax2.set_title('Cross-Validation Scores with Standard Deviation')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("Model comparison visualization saved as 'model_comparison.png'")
    plt.show()
    
    return results

def main():
    """Main function to run all scikit-learn demonstrations"""
    print("🔬 Scikit-learn Fundamentals")
    print("=" * 50)
    
    # API overview
    scikit_learn_api_overview()
    
    # Basic algorithms
    basic_supervised_algorithms()
    
    # Cross-validation
    cross_validation_demo()
    
    # Hyperparameter tuning
    hyperparameter_tuning()
    
    # Pipeline creation
    pipeline_creation()
    
    # Model comparison
    model_comparison()
    
    print("\n" + "=" * 50)
    print("✅ Lesson 5 Complete!")
    print("Next: Learn linear regression and regression techniques")
    print("Key takeaway: Scikit-learn provides a consistent API for all ML algorithms")

if __name__ == "__main__":
    main()
